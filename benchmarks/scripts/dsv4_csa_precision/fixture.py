# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#!/usr/bin/env python3
"""Single-device native CSA impl.forward versus the native-layout PTO entry.

Only one checkpoint attention module is loaded, not a model or serving engine.
Cache allocation snapshots are independent between the two execution paths.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def initialize(args, stack):
    # Accuracy comparisons require deterministic native operators and HCCL.
    os.environ["HCCL_DETERMINISTIC"] = "true"
    # scikit-build editable finder precedes PYTHONPATH; select only the build SDK
    # from the exact clean checkout, leaving installed simpler binaries untouched.
    from importlib.machinery import PathFinder

    sdk_root = str(args.sdk_root)

    class SDKFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == "simpler_setup":
                return PathFinder.find_spec(fullname, [sdk_root])
            if fullname.startswith("simpler_setup."):
                return PathFinder.find_spec(fullname, path)
            return None

    sys.meta_path.insert(0, SDKFinder)

    import torch
    import torch_npu  # noqa: F401 -- Registers torch.npu before selecting the device.

    import vllm_ascend
    import vllm_ascend.attention

    vllm_ascend.attention.__path__.insert(0, str(args.variant_dir))

    torch.set_num_threads(4)
    torch.manual_seed(62)
    vllm_ascend.__path__.append(str(args.extension_dir.resolve()))
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("Native vLLM Ascend operators could not be loaded")
    torch.npu.set_device(args.device)
    torch_npu.npu.set_deterministic_level(1)
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import adapt_patch, register_ascend_customop

    engine_args = EngineArgs(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=max(4096, args.start_pos + args.seq * args.steps + 128),
        max_num_seqs=max(4, args.batch),
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        block_size=128,
        quantization="ascend",
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={
            "weight_nz_mode": 0,
            "multistream_dsv4_dsa_overlap": False,
        },
    )
    cfg = engine_args.create_engine_config()
    stack.enter_context(set_current_vllm_config(cfg))
    init_ascend_config(cfg)
    register_ascend_customop(cfg)
    adapt_patch()
    rendezvous = Path(tempfile.mkdtemp(prefix="tp1-", dir=args.out_dir)) / "rendezvous"
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=args.device,
        distributed_init_method=rendezvous.as_uri(),
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="hccl")
    deterministic_level = torch_npu.npu._get_deterministic_level()
    assert deterministic_level == 1, f"Unexpected deterministic level: {deterministic_level}"
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    print("DETERMINISM_VERIFIED level=1 HCCL_DETERMINISTIC=true", flush=True)
    return cfg


def load_attention(args, cfg):
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention

    quant_cfg = cfg.quant_config
    hf = cfg.model_config.hf_config
    with torch.device(f"npu:{args.device}"):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            attention = DeepseekV4Attention(
                cfg,
                hf,
                max_position_embeddings=hf.rope_parameters["original_max_position_embeddings"],
                cache_config=cfg.cache_config,
                quant_config=quant_cfg,
                prefix=f"model.layers.{args.layer}.self_attn",
            )
        finally:
            torch.set_default_dtype(previous_dtype)

    parameters = dict(attention.named_parameters())
    prefix = f"layers.{args.layer}.attn."
    weight_map = json.loads((args.model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    loaded = set()
    for shard in sorted(set(selected.values())):
        with safe_open(str(args.model / shard), framework="pt", device="cpu") as handle:
            for source, source_shard in selected.items():
                if source_shard != shard:
                    continue
                name = source[len(prefix) :]
                if name.endswith(".scale"):
                    name = name[:-6] + ".weight_scale"
                if name not in parameters:
                    raise KeyError(f"Checkpoint parameter {source} does not map to attention: {name}")
                param = parameters[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if name == "wo_a.weight":
                    # Use Ascend's grouped-output loader; the generic V2 loader
                    # attached by compressed-tensors only fills the flat weight.
                    loader = attention.wo_a.weight_loader
                weight = handle.get_tensor(source)
                # The checkpoint stores per-channel scales flat, whereas the
                # native compressed-tensors loader allocates [out_features, 1].
                if name.endswith(".weight_scale") and weight.numel() == param.numel():
                    weight = weight.reshape(param.shape)
                print(f"[weights] {name}: {tuple(weight.shape)} -> {tuple(param.shape)}", flush=True)
                loader(param, weight)
                loaded.add(name)
    # Quantization-specific offset buffers can be initialized by the quantizer;
    # every mathematical weight, norm and APE must have a checkpoint source.
    missing = set(parameters) - loaded
    if any(not name.endswith("weight_offset") for name in missing):
        raise RuntimeError(f"Unloaded attention parameters: {sorted(missing)}")
    for name in missing:
        parameters[name].data.zero_()
    for module in attention.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
    return attention, {
        "checkpoint": str(args.model),
        "layer": args.layer,
        "loaded": sorted(loaded),
        "initialized_auxiliary": sorted(missing),
        "parameter_shapes": {name: [list(p.shape), str(p.dtype)] for name, p in attention.named_parameters()},
    }


def cache_views(roots):
    """Recreate vLLM's six views without losing shared-allocation aliasing."""
    import torch

    main, raw, index = roots
    main_state = main.as_strided((main.shape[0], 8, 1, 2048), (32768, 2048, 2048, 1))
    cmp_kv = main.view(torch.bfloat16).view(-1, 128, 1, 512)
    inner_state = index.view(torch.float32).as_strided(
        (index.shape[0], 8, 1, 512),
        (4160, 512, 512, 1),
    )
    key = index.as_strided((index.shape[0], 128, 1, 128), (16640, 128, 128, 1))
    scale = index.view(torch.float16).as_strided(
        (index.shape[0], 128, 1, 1),
        (8320, 1, 1, 1),
        storage_offset=8192,
    )
    return cmp_kv, raw, main_state, inner_state, key, scale


def build_payload(args):
    """Use absolute native page tables and disjoint writable request pages.

    Weights are real; hidden states and historical caches are reproducible
    synthetic inputs. This is an actual impl.forward test, not a full request.
    """
    import torch

    first, end = args.start_pos, args.start_pos + args.seq * args.steps
    state = torch.zeros(args.batch, (end + 7) // 8, dtype=torch.int32)
    raw = torch.zeros(args.batch, (end + 127) // 128, dtype=torch.int32)
    cmp = torch.zeros(args.batch, (end + 511) // 512, dtype=torch.int32)
    next_page, next_raw = 1, 1
    state_pages, cmp_pages = [], []
    for request in range(args.batch):
        for col in range(cmp.shape[1]):
            cmp[request, col] = next_page
            cmp_pages.append(next_page)
            next_page += 1
        for col in range(max(0, first - 7) // 8, state.shape[1]):
            state[request, col] = next_page
            state_pages.append(next_page)
            next_page += 1
        for col in range(max(0, first - 127) // 128, raw.shape[1]):
            raw[request, col] = next_raw
            next_raw += 1
    roots = (
        torch.zeros(next_page, 16, 2048, dtype=torch.float32),
        torch.randn(next_raw, 128, 1, 512, dtype=torch.bfloat16) * 0.1,
        torch.zeros(next_page, 130, 128, dtype=torch.int8),
    )
    cmp_kv, _, main_state, inner_state, key, scale = cache_views(roots)
    for page in cmp_pages:
        cmp_kv[page].copy_(torch.randn_like(cmp_kv[page]) * 0.1)
        key[page].copy_(torch.randint(-100, 101, key[page].shape, dtype=torch.int8))
        scale[page].fill_(0.01)
    for page in state_pages:
        main_state[page].copy_(torch.randn_like(main_state[page]) * 0.1)
        inner_state[page].copy_(torch.randn_like(inner_state[page]) * 0.1)
    device = f"npu:{args.device}"
    native_roots = tuple(t.to(device) for t in roots)
    pto_roots = tuple(t.to(device) for t in roots)
    tables_cpu = (cmp, state, state.clone(), cmp.clone(), raw)
    # The public output exposes only real token rows. Guard the next tile so a
    # missing tail mask fails explicitly instead of corrupting another input.
    pto_out_storage = torch.full(
        (args.batch * args.seq + 8, 4096),
        -12.5,
        dtype=torch.bfloat16,
        device=device,
    )
    return {
        "initial": roots,
        "native_roots": native_roots,
        "pto_roots": pto_roots,
        "native": cache_views(native_roots),
        "pto": cache_views(pto_roots),
        "tables_cpu": tables_cpu,
        "tables": tuple(t.to(device) for t in tables_cpu),
        "state_pages": state_pages,
        "cmp_pages": cmp_pages,
        "x": torch.randn(args.batch * args.seq, 4096, dtype=torch.bfloat16).to(device),
        "native_out": torch.empty(args.batch * args.seq, 4096, dtype=torch.bfloat16, device=device),
        "pto_out": pto_out_storage[: args.batch * args.seq],
        "pto_output_guard": pto_out_storage[args.batch * args.seq :],
    }


def build_metadata(args, payload, start_pos):
    import torch

    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata, AscendDSAReqMetadata
    from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa, get_full_cos_and_sin_dsa

    device = payload["x"].device
    b, s = args.batch, args.seq
    t, end = b * s, start_pos + s
    pos_cpu = torch.arange(start_pos, end, dtype=torch.int64).repeat(b)
    pos = pos_cpu.to(device)
    qstarts_cpu = torch.arange(0, t + 1, s, dtype=torch.int32)
    qstarts = qstarts_cpu.to(device)
    lengths = torch.full((b,), end, dtype=torch.int32, device=device)
    starts = torch.full((b,), start_pos, dtype=torch.int32, device=device)
    cos, sin = get_cos_and_sin_dsa(pos)
    boundaries = (pos_cpu + 1) % 4 == 0
    cmp_cos, cmp_sin = get_full_cos_and_sin_dsa("c4")
    sas = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=qstarts,
        seqused_kv=lengths,
        max_seqlen_q=s,
        max_seqlen_kv=end,
        batch_size=b,
        cmp_topk=512,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device=str(device),
    )
    from vllm_ascend.device.device_op import DeviceOperator

    qli_lengths = torch.div(lengths, 4, rounding_mode="floor")
    qli_residual = lengths % 4
    qli = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        quant_mode=DeviceOperator.get_dsa_indexer_quant_mode(),
        cu_seqlens_q=qstarts,
        seqused_k=qli_lengths,
        cmp_residual_k=qli_residual,
        batch_size=b,
        max_seqlen_q=s,
        max_seqlen_k=end // 4,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device=str(device),
    )
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0)
    hadamard = hadamard.to(device=device, dtype=torch.bfloat16)
    metadata = []
    request_ids = torch.arange(b).repeat_interleave(s)
    for group, (table_cpu, table) in enumerate(zip(payload["tables_cpu"], payload["tables"])):
        rows = pos_cpu // 4 if group in (0, 3) else pos_cpu
        page_size = 8 if group in (1, 2) else 128
        # AscendDSAMetadataBuilder converts flat allocator slots into native
        # scatter_nd coordinates. A flat [N] tensor instead means ONE N-axis
        # index to the scatter operator and can write outside the intended row.
        slots = torch.stack(
            (table_cpu[request_ids, rows // page_size].long(), rows % page_size),
            dim=-1,
        )
        if group in (0, 3):
            slots = slots[boundaries]
        slots = slots.to(device=device, dtype=torch.int64)
        req = AscendDSAReqMetadata(
            input_positions=pos,
            block_table=table,
            seq_lens=lengths,
            slot_mapping=slots.to(torch.int32),
            storage_block_size=page_size,
            query_start_loc=qstarts,
            sin=sin,
            cos=cos,
            full_compress_sin=cmp_sin,
            full_compress_cos=cmp_cos,
            num_compressed_tokens=min(t, t // 4 + b),
            num_actual_reqs=b,
            cache_group_key=f"ab-group-{group}",
            start_pos=starts,
            sas_metadata=sas,
            qli_metadata=qli,
            qli_cu_seqlens_q=qstarts,
            qli_seqused_k=qli_lengths,
            qli_cmp_residual_k=qli_residual,
            ori_win_left=127,
            ori_win_right=0,
        )
        metadata.append(
            AscendDSAMetadata(
                num_actual_tokens=t,
                num_decodes=b,
                num_decode_tokens=t,
                num_prefills=0,
                req_metadata=req,
                hadamard=hadamard,
            )
        )
    return metadata


def difference(actual, expected):
    import torch

    a, e = actual.detach().float().cpu().reshape(-1), expected.detach().float().cpu().reshape(-1)
    diff = (a - e).abs()
    relative = diff / torch.maximum(a.abs(), e.abs()).clamp_min(1e-3)
    # Large unchanged cache pools need wide reductions: FP32 accumulation can
    # otherwise report a cosine above one even for identical BF16 histories.
    a_sq = a.square().sum(dtype=torch.float64)
    e_sq = e.square().sum(dtype=torch.float64)
    d_sq = diff.square().sum(dtype=torch.float64)
    cosine = (a * e).sum(dtype=torch.float64) / (a_sq * e_sq).sqrt().clamp_min(1e-24)
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
        "shape": list(actual.shape),
        "max_abs": diff.max().item(),
        "different_elements": int((a != e).count_nonzero()),
        "mean_abs": diff.sum(dtype=torch.float64).item() / diff.numel(),
        "relative_l2": (d_sq.sqrt() / e_sq.sqrt().clamp_min(1e-12)).item(),
        "cosine": cosine.item(),
        "frac_rdiff_gt_5e-3": (relative > 5e-3).count_nonzero().item() / relative.numel(),
        "frac_rdiff_gt_1e-2": (relative > 1e-2).count_nonzero().item() / relative.numel(),
        "allclose_1e-2": bool(torch.allclose(a, e, rtol=1e-2, atol=1e-2)),
    }
