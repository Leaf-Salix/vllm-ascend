# SPDX-License-Identifier: Apache-2.0
"""当前 release 的单卡 Native/PTO 整层诊断；不加载 MoE，不代替整模型验收。"""

import argparse
import importlib
import json
import math
import os
import statistics
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_validation import compare_tensor, compare_topk


def writable_bytes(allocation, views, slots):
    """只允许写 Native slot 对应的逻辑行；页内 padding 与首尾保护区仍须保持原值。"""
    import torch

    allowed = torch.zeros(allocation.numel(), dtype=torch.bool)
    for page, row in slots.detach().cpu().tolist():
        if page < 0 or row < 0:
            continue
        for view in views:
            if page >= view.shape[0] or row >= view.shape[1]:
                raise ValueError(f"slot 超出 Native 页视图：{page}, {row}, {view.shape}")
            begin = (view.storage_offset() + page * view.stride(0) + row * view.stride(1)) * view.element_size()
            end = begin + view.shape[-1] * view.element_size()
            if end > allowed.numel():
                raise ValueError("slot 写区超出保护区分配")
            allowed[begin:end] = True
    return allowed


def make_layer(config, checkpoint, device, layer_index=2):
    import torch
    from dsv4_csa_formal_weights import load_formal_layer_weights
    from safetensors import safe_open
    from vllm.model_executor.layers.layernorm import RMSNorm

    from vllm_ascend.models.deepseek_v4 import DeepseekV2DecoderLayer, DeepseekV4Attention
    from vllm_ascend.utils import register_ascend_customop

    register_ascend_customop(config)
    hf = config.model_config.hf_config

    class AttentionHalf(DeepseekV2DecoderLayer):
        def __init__(self):
            # 复用 Native HC 方法，仅构造 attention 半层，不分配 MoE 权重。
            torch.nn.Module.__init__(self)
            self.hc_mult, self.hc_sinkhorn_iters = hf.hc_mult, hf.hc_sinkhorn_iters
            self.norm_eps, self.hc_eps = hf.rms_norm_eps, hf.hc_eps
            self.self_attn = DeepseekV4Attention(
                config,
                hf,
                max_position_embeddings=hf.rope_parameters["original_max_position_embeddings"],
                cache_config=config.cache_config,
                quant_config=config.quant_config,
                prefix=f"model.layers.{layer_index}.self_attn",
            )
            self.input_layernorm = RMSNorm(hf.hidden_size, eps=hf.rms_norm_eps)

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            layer = AttentionHalf()
    finally:
        torch.set_default_dtype(old_dtype)
    records, methods = load_formal_layer_weights(layer.self_attn, checkpoint, layer_index)
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm.weight"):
        key = f"layers.{layer_index}.{name}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
            value = reader.get_tensor(key)
        if name == "attn_norm.weight":
            layer.input_layernorm.weight.data.copy_(value)
        else:
            setattr(layer, name, torch.nn.Parameter(value.to(device)))
        records.append({"name": key, "shard": index[key], "shape": list(value.shape), "dtype": str(value.dtype)})
    layer.self_attn.dsa_attn._pto_csa_layer = (layer,)
    return layer, {"weights": records, "quant_methods": methods}


def make_fixture(config, attention, batch, history, seed, device):
    import torch
    from dsv4_csa_native_case import allocate_native_cache, make_cache_groups

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    tokens = batch * 6
    bounds_cpu = torch.arange(batch + 1, dtype=torch.int32) * 6
    lengths_cpu = torch.full((batch,), history + 6, dtype=torch.int32)
    bounds, lengths = bounds_cpu.to(device), lengths_cpu.to(device)
    positions = (history + torch.arange(6)).repeat(batch).to(device=device, dtype=torch.int64)
    hidden = torch.randn((tokens, 4, 4096), device=device, dtype=torch.bfloat16)
    groups = make_cache_groups(config, device, attention)
    metadata, common_cache, prefill_cache, decode_cache = {}, {}, {}, {}
    for name, group in groups.items():
        spec = group["spec"]
        ratio = getattr(spec, "compress_ratio", 1)
        columns = (history + 6 + spec.block_size * ratio - 1) // (spec.block_size * ratio) + 1
        # 非压缩历史仅需保留滑窗/近期 state；页表保留完整逻辑列并循环映射独占物理页。
        per_request = columns if name in ("compressed", "indexer") else 9
        pages = batch * per_request + 1
        group["layout"] = allocate_native_cache(group, pages, device)
        for i, view in enumerate(group["views"]):
            if view.dtype == torch.int8:
                view.random_(-64, 64)
            elif name == "indexer" and i == 1:
                view.fill_(0.01)
            else:
                view.normal_(0, 0.5)
        group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
        table = BlockTable(spec.block_size, 40, columns, 256, True, device, num_speculative_tokens=5)
        for row in range(batch):
            table.add_row(
                [1 + row * per_request + (per_request - 1 - col) % per_request for col in range(columns)], row
            )
        table.commit_block_table(batch)
        table.compute_slot_mapping(batch, bounds, positions // ratio)
        common = AscendCommonAttentionMetadata(
            query_start_loc=bounds,
            query_start_loc_cpu=bounds_cpu,
            seq_lens=lengths,
            _seq_lens_cpu=lengths_cpu,
            seq_lens_cpu=lengths_cpu,
            num_reqs=batch,
            num_actual_tokens=tokens,
            num_input_tokens=tokens,
            max_query_len=6,
            max_seq_len=history + 6,
            block_table_tensor=table.block_table.gpu[:batch],
            slot_mapping=table.slot_mapping.gpu[:tokens],
            positions=positions,
            attn_state=AscendAttentionState.SpecDecoding,
            causal=True,
        )
        metadata[group["prefix"]] = group["builder"].build(
            0,
            common,
            num_reqs_actual=batch,
            block_size=spec.block_size,
            common_ratio_to_sas_metadata=common_cache,
            prefill_ratio_to_sas_metadata=prefill_cache,
            decode_ratio_to_sas_metadata=decode_cache,
        )
        group.update(table=table, common=common)
    compact = {
        name: attention.dsa_attn.dsa_attn.impl._compute_compressor_metadata(metadata[groups[name]["prefix"]].decode)
        for name in ("compressed", "indexer")
    }
    for name, group in groups.items():
        slots = compact[name][2] if name in compact else metadata[group["prefix"]].decode.slot_mapping
        group["allowed"] = writable_bytes(group["allocation"], group["views"], slots)
        group["initial"] = group["allocation"].cpu()
    readonly = {"positions": positions}
    for name, group in groups.items():
        req = metadata[group["prefix"]].decode
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
            value = getattr(req, field)
            if value is not None:
                readonly[f"{name}.{field}"] = value
    readonly.update({f"{name}.compact_slots": value[2] for name, value in compact.items()})
    readonly = {name: (value, value.cpu()) for name, value in readonly.items()}
    return {
        "groups": groups,
        "metadata": metadata,
        "compact": compact,
        "positions": positions,
        "readonly": readonly,
        "hidden": hidden,
        "tokens": tokens,
    }


def collect_state(fixture, output, topk):
    state = {"x_out": output.detach().cpu(), "idx_topk": topk.detach().cpu().reshape(fixture["tokens"], 512)}
    for name, group in fixture["groups"].items():
        for index, view in enumerate(group["views"]):
            state[f"{name}.{index}"] = view.detach().cpu()
    return state


def guard_checks(fixture):
    result = {}
    for name, group in fixture["groups"].items():
        changed = group["allocation"].cpu() != group["initial"]
        bad = changed & ~group["allowed"]
        result[name] = {
            "status": "FAIL" if bool(bad.any()) else "PASS",
            "changed_bytes": int(changed.sum()),
            "outside_slot_bytes": int(bad.sum()),
            "first_outside": bad.nonzero()[:8].flatten().tolist(),
        }
    for name, (value, initial) in fixture.get("readonly", {}).items():
        result[f"metadata.{name}"] = compare_tensor(value, initial, 0, 0)
    return result


def restore(fixture):
    for group in fixture["groups"].values():
        group["allocation"].copy_(group["initial"])


def check_graph_replay(fixture, call, eager_a, report):
    """同一组地址更新输入 A→B→A，检查图输出、状态及保护区；固定规约下精确比较。"""
    import torch

    hidden = fixture["hidden"]
    input_a = hidden.clone()
    input_b = -input_a
    result = {"status": "RUNNING", "scope": "单卡固定形状/metadata 的输入内容更新；不代表 padding/整模型图验收"}
    report["graph"] = result
    try:
        hidden.copy_(input_b)
        restore(fixture)
        call()
        torch.npu.synchronize()
        eager_b = collect_state(fixture, call.args["x_out"], call.args["idx_topk"])
        if torch.equal(eager_a["x_out"], eager_b["x_out"]):
            raise ValueError("图测试的 A/B 输出相同，不能验证输入更新")
        hidden.copy_(input_a)
        restore(fixture)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call()
        torch.npu.synchronize()
        result["replays"] = []
        for name, value, reference in (("A", input_a, eager_a), ("B", input_b, eager_b), ("A", input_a, eager_a)):
            hidden.copy_(value)
            restore(fixture)
            graph.replay()
            torch.npu.synchronize()
            actual = collect_state(fixture, call.args["x_out"], call.args["idx_topk"])
            checks = {key: compare_tensor(actual[key], expected, 0, 0) for key, expected in reference.items()}
            guards = guard_checks(fixture)
            result["replays"].append({"input": name, "eager_comparison": checks, "guards": guards})
            if any(check["status"] != "PASS" for check in (*checks.values(), *guards.values())):
                raise ValueError(f"图重放 {name} 与相同输入的 eager 不一致，或改写保护区/metadata")
        result["status"] = "PASS"
    except BaseException:
        result["status"] = "FAIL"
        raise
    finally:
        hidden.copy_(input_a)
        restore(fixture)


def check_padding_graph(fixture, call, eager, make_call, impl, report):
    """Native builder 更新同址输入；捕获的 metadata producer 和 PTO 图随有效请求数变化。"""
    import torch

    batch = fixture["tokens"] // 6
    groups = fixture["groups"]
    originals = {
        name: {
            "slots": group["common"].slot_mapping.clone(),
            "table": group["common"].block_table_tensor.clone(),
            "lengths": group["common"]._seq_lens_cpu.clone(),
            "allowed": group["allowed"],
        }
        for name, group in groups.items()
    }
    original_readonly = fixture["readonly"]
    result = {
        "status": "RUNNING", "bucket_batch": batch,
        "scope": "单卡 PTO 同一图的满档→补位→满档；Native builder 在图外更新、"
                 "Native compact producer 在图内执行；不代表 Native 整图或空 rank 验收",
        "padding": "seq_lens=0、slot=-1、页表=0；positions 与尾部 RoPE 保留旧值",
        "replays": [],
    }
    report["padding_graph"] = result
    captured = {}

    def run():
        # 与生产图一样，捕获时 num_reqs_actual/输出容量固定；device 输入在重放前更新。
        compact = {
            name: impl._compute_compressor_metadata(fixture["metadata"][groups[name]["prefix"]].decode)
            for name in ("compressed", "indexer")
        }
        captured.update(compact)
        make_call(compact)()

    def update_metadata(active):
        common_cache, prefill_cache, decode_cache, metadata = {}, {}, {}, {}
        for name, group in groups.items():
            common, original = group["common"], originals[name]
            common._seq_lens_cpu.copy_(original["lengths"])
            common._seq_lens_cpu[active:].zero_()
            common.seq_lens.copy_(common._seq_lens_cpu)
            common.num_actual_tokens = active * 6
            common.slot_mapping.copy_(original["slots"])
            common.slot_mapping[active * 6:].fill_(-1)
            common.block_table_tensor.copy_(original["table"])
            common.block_table_tensor[active:].zero_()
            current = group["builder"].build(
                0, common, num_reqs_actual=active, block_size=group["spec"].block_size,
                common_ratio_to_sas_metadata=common_cache,
                prefill_ratio_to_sas_metadata=prefill_cache,
                decode_ratio_to_sas_metadata=decode_cache,
            )
            old = fixture["metadata"][group["prefix"]].decode
            for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
                before, after = getattr(old, field), getattr(current.decode, field)
                if before is not None and (after is None or before.data_ptr() != after.data_ptr()):
                    raise ValueError(f"Native builder 替换了捕获输入地址：{name}.{field}")
            metadata[name] = current
        return metadata

    try:
        restore(fixture)
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        counts = [batch, *dict.fromkeys((batch - 1, 1)), batch]
        for active in counts:
            metadata = update_metadata(active)
            # 按实际请求数生成独立的 Native oracle，只用其有效行规定写区与期望 metadata。
            oracle = {name: impl._compute_compressor_metadata(metadata[name].decode)
                      for name in ("compressed", "indexer")}
            expected = {name: eager[name][:active * 6] for name in ("x_out", "idx_topk")}
            compact_checks = {}
            for name, group in groups.items():
                slots = oracle[name][2] if name in oracle else metadata[name].decode.slot_mapping
                slots_cpu = slots.cpu()
                group["allowed"] = writable_bytes(group["allocation"], group["views"], slots)
                for index, view in enumerate(group["views"]):
                    key = f"{name}.{index}"
                    initial = group["initial"].view(view.dtype).as_strided(
                        view.shape, view.stride(), view.storage_offset()).clone()
                    for page, row in slots_cpu.tolist():
                        if page >= 0 and row >= 0:
                            initial[page, row] = eager[key][page, row]
                    expected[key] = initial
            fixture["readonly"] = {name: (value, value.cpu()) for name, (value, _) in original_readonly.items()}
            restore(fixture)
            call.args["x_out"].fill_(float("nan"))
            call.args["idx_topk"].fill_(-12345)
            graph.replay()
            torch.npu.synchronize()
            actual = collect_state(fixture, call.args["x_out"], call.args["idx_topk"])
            actual.update({name: actual[name][:active * 6] for name in ("x_out", "idx_topk")})
            for name, values in oracle.items():
                valid = (values[2].cpu() >= 0).all(dim=1)
                rows = valid.nonzero().flatten()
                for field, value, reference in zip(("cos", "sin", "slots"), captured[name], values):
                    compact_checks[f"{name}.{field}"] = compare_tensor(
                        value.cpu()[rows], reference.cpu()[rows], 0, 0)
            checks = {key: compare_tensor(actual[key], value, 0, 0) for key, value in expected.items()}
            guards = guard_checks(fixture)
            result["replays"].append({
                "active_batch": active, "reference": "同一实现满档有效请求前缀；补位 cache/state 保持初态",
                "state_comparison": checks, "compact_metadata": compact_checks, "guards": guards,
            })
            all_checks = (*checks.values(), *compact_checks.values(), *guards.values())
            if any(value["status"] != "PASS" for value in all_checks):
                raise ValueError(f"有效请求数 {active}/{batch} 的图重放输出、metadata 或保护区失败")
        result["status"] = "PASS"
    except BaseException:
        result["status"] = "FAIL"
        raise
    finally:
        update_metadata(batch)
        for name, group in groups.items():
            group["allowed"] = originals[name]["allowed"]
        fixture["readonly"] = original_readonly
        restore(fixture)


def measure_graph_interval(fixture, run, output, topk, reference, *, iters, warmup, require_exact,
                           profile_dir=None):
    """图外事件包住一次整层重放；初态恢复与输出毒化均在计时区间外。"""
    import torch

    initial = {
        name: group["initial"].to(group["allocation"].device)
        for name, group in fixture["groups"].items()
    }

    def reset():
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(initial[name])
        output.fill_(float("nan"))

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    # 当前 torch_npu 图内 Event 不随重放更新时间戳，必须在图外显式 record。
    start.record()
    end.record()
    reset()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    samples, timestamps = [], []
    for iteration in range(warmup + iters):
        reset()
        start.record()
        graph.replay()
        end.record()
        torch.npu.synchronize()
        stamp = start.recorded_time()
        if timestamps and stamp <= timestamps[-1]:
            raise ValueError("图计时的开始事件时间戳未更新，不能采纳重复的旧时间")
        timestamps.append(stamp)
        if iteration >= warmup:
            elapsed = start.elapsed_time(end) * 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"图外设备事件没有产生有效时间戳：{elapsed}")
            samples.append(elapsed)
    state = collect_state(fixture, output, topk())
    checks = {name: compare_tensor(value, reference[name], 0, 0) for name, value in state.items()}
    guards = guard_checks(fixture)
    if any(check["status"] != "PASS" for check in guards.values()):
        raise ValueError("计时图重放改写了 metadata 或 slot 外存储")
    if any(check.get("nonfinite") != 0 for check in checks.values()):
        raise ValueError("计时图重放出现非有限值或不完整输出")
    if require_exact and any(check["status"] != "PASS" for check in checks.values()):
        raise ValueError("固定规约的计时图与同初态 eager 不一致")
    selection = compare_topk(state["idx_topk"], reference["idx_topk"], (fixture["positions"].cpu() + 1) // 4)
    if selection["status"] == "FAIL":
        raise ValueError("计时图 Top-K 含越界、重复或缺失的索引")
    profile = None
    if profile_dir is not None:
        import torch_npu

        reset()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        ) as trace:
            start.record()
            graph.replay()
            end.record()
            torch.npu.synchronize()
            trace.step()
        profile = {"directory": str(profile_dir), "event_envelope_us": start.elapsed_time(end) * 1000,
                   "scope": "独立一次图重放，核对设备区间和热点；不混入无 profiler 的采样"}
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        # recorded_time 的原始计数只用于检查更新；耗时单位由 elapsed_time 给出。
        "start_timestamps_raw": timestamps[warmup:],
        "us_min": ordered[0], "us_p50": statistics.median(samples),
        "us_p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "us_max": ordered[-1],
        "eager_comparison": checks, "exact_comparison_required": require_exact,
        "topk_selection": selection, "guards": guards,
        "profile": profile,
    }


def run(args, report):
    report["execution_config"] = {
        "path": "manual_npu_graph",
        "enable_npugraph_ex": False,
        "enable_static_kernel": False,
        "worker_cpu_binding": False,
        "scope": "核内与手工图区间诊断；未经过decode模板的编译优化，不作为该模板下Native性能验收",
    }
    activate()
    import torch
    import torch_npu
    from dsv4_csa_native_case import native_session
    from dsv4_csa_replay import argument_roles, capture_tensors, save_snapshot
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    current_platform.pre_register_and_update()
    torch.npu.set_device(args.device)
    # 与 Native NPUModelRunner 一致：必须在权重后处理前启用，否则 NZ 转换静默退回 ND。
    torch.npu.config.allow_internal_format = True
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("Native 自定义算子未完成注册")
    torch_npu.npu.set_deterministic_level(args.deterministic_level)
    config = EngineArgs(
        model=str(args.checkpoint),
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        quantization="ascend",
        hf_overrides={"sliding_window": 128},
        max_model_len=max(16384, args.history + 128),
        max_num_seqs=40,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enforce_eager=True,
        block_size=32,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={"weight_nz_mode": args.weight_nz_mode, "enable_kv_nz": False, "enable_dsa_cp": False},
    ).create_engine_config()
    device = torch.device(f"npu:{args.device}")
    with native_session(config, args.device), torch.inference_mode():
        layer, details = make_layer(config, args.checkpoint, device, args.layer_index)
        report.update(details)
        fixture = make_fixture(config, layer.self_attn, args.batch, args.history, args.seed, device)
        report["layouts"] = {name: group["layout"] for name, group in fixture["groups"].items()}
        from vllm_ascend.ascend_config import get_ascend_config

        report["effective_weight_nz_mode"] = get_ascend_config().weight_nz_mode
        report["native_weight_formats"] = {
            name: torch_npu.get_npu_format(dict(layer.self_attn.named_parameters())[f"{name}.weight"])
            for name in ("wq_a", "wq_b", "wo_a", "wo_b")
        }
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, _should_trans_nz

        expected_formats = {
            name: "NZ" if not getattr(getattr(layer.self_attn, name), "keep_weight_nd", False)
            and _should_trans_nz(dict(layer.self_attn.named_parameters())[f"{name}.weight"]) else "ND"
            for name in report["native_weight_formats"]
        }
        report["native_expected_layouts"] = expected_formats
        for name, expected in expected_formats.items():
            actual = report["native_weight_formats"][name]
            if actual not in ((ACL_FORMAT_FRACTAL_NZ,) if expected == "NZ" else (0, 2)):
                raise ValueError(f"Native 权重实际格式与 mode 不一致：{name} expected={expected}, actual={actual}")
        report["native_compressor_weight_formats"] = {
            f"{prefix}.{name}": torch_npu.get_npu_format(getattr(compressor, name).weight)
            for prefix, compressor in (("compressor", layer.self_attn.compressor),
                                       ("indexer.compressor", layer.self_attn.indexer.compressor))
            for name in ("wkv", "wgate")
        }
        if any(value not in (0, 2) for value in report["native_compressor_weight_formats"].values()):
            raise ValueError("Native 融合 Compressor 的 wkv/wgate 必须遵循 ND 入参合同")
        output = torch.empty_like(fixture["hidden"])
        original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
        original_sparse = torch.ops._C_ascend.npu_sparse_attn_sharedkv
        captured = {}

        def record_sparse(q, **kwargs):
            value = original_sparse(q, **kwargs)
            if args.save_sparse_case and "sparse_case" not in captured:
                # 在 Native 逆 RoPE 原地修改输出之前保存，用于隔离 QK/softmax/PV。
                payload = {name: kwargs[name].detach().cpu() for name in (
                    "ori_kv", "cmp_kv", "ori_block_table", "cmp_block_table",
                    "cmp_sparse_indices", "seqused_kv", "sinks",
                )}
                payload.update(q=q.detach().cpu(), expected=value[0].detach().cpu(),
                               position_ids=fixture["positions"].detach().cpu(),
                               batch=args.batch, history=args.history)
                torch.save(payload, args.output / "native_sparse.pt")
                captured["sparse_case"] = True
            return value

        def record_qli(*inputs, **kwargs):
            value = original_qli(*inputs, **kwargs)
            captured["topk"] = value[0].detach().clone()
            if args.save_case and not native:
                # Native 不返回 score；保留其真实 QLI 输入，后续可离线分析选择边界。
                captured["indexer_inputs"] = {
                    name: kwargs[name].detach().cpu()
                    for name in ("query", "weights", "query_dequant_scale", "block_table")
                }
            return value

        torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_qli
        if args.save_sparse_case:
            torch.ops._C_ascend.npu_sparse_attn_sharedkv = record_sparse
        native = []
        report["native_guards"] = []
        try:
            for _ in range(2):
                restore(fixture)
                with set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
                ):
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)
                torch.npu.synchronize()
                native.append(collect_state(fixture, output, captured["topk"]))
                report["native_guards"].append(guard_checks(fixture))
        finally:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
            torch.ops._C_ascend.npu_sparse_attn_sharedkv = original_sparse
        if args.save_sparse_case:
            if not captured.get("sparse_case"):
                raise RuntimeError("未捕获到 Native sparse attention 调用")
            report["saved_sparse_case"] = "native_sparse.pt"
        report["native_self"] = {
            name: compare_tensor(native[1][name], value, 0, 0) for name, value in native[0].items()
        }

        if args.native_profile_only:
            # 补 Native 分算子 trace 缺口，不为此编译/执行 PTO 或重复整套矩阵。
            def profile_qli(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                captured["profile_topk"] = value[0]
                return value

            def profile_native_call():
                with set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
                ):
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)

            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = profile_qli
            try:
                report["timing"] = {"native": measure_graph_interval(
                    fixture, profile_native_call, output, lambda: captured["profile_topk"], native[0],
                    iters=args.timing_iters, warmup=args.timing_warmup,
                    require_exact=bool(args.deterministic_level), profile_dir=args.output / "profile/native",
                )}
            finally:
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
            report.update(status="MEASURED", scope="仅 Native 图重放与分算子 trace；不含 PTO 对照或整模型验收")
            return

        import pypto.torch

        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.nz_mode import root_weight_layouts
        from vllm_ascend.ops.pypto.variant import variant_package

        package = variant_package()
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import ATOMIC_ADD

        reduction = importlib.import_module(f"{package}.qkv_proj_rope")
        report["pto_reduction"] = {
            "atomic_add": ATOMIC_ADD, "qr_split_k": reduction.QR_OK, "kv_split_k": reduction.KV_OK,
        }
        if (args.graph or args.padding_graph) and ATOMIC_ADD:
            raise ValueError("图正确性检查使用逐元素精确比较，须设置 --atomic-add 0 排除跨核规约波动")
        adapter = importlib.import_module(f"{package}.native_adapter")
        module = importlib.import_module(f"{package}.decode_csa")
        root = module._decode_csa_tp1_layer
        layouts = root_weight_layouts(root)
        hadamard = fixture["metadata"][fixture["groups"]["indexer"]["prefix"]].hadamard
        weights = adapter.prepare_weights(layer.self_attn, hadamard, layer)
        from pypto._torch_npu import storage_shape

        report["weight_storage_binding"] = {}
        for name, layout in layouts.items():
            original = getattr(layer.self_attn, name).weight
            prepared = weights[name]
            source_format = int(torch_npu.get_npu_format(original))
            reused = prepared.data_ptr() == original.data_ptr()
            report["weight_storage_binding"][name] = {
                "shape": list(prepared.shape), "native_format": source_format,
                "native_shape": list(original.shape), "native_stride": list(original.stride()),
                "native_storage_shape": storage_shape(original),
                "pto_format": int(torch_npu.get_npu_format(prepared)),
                "same_data_ptr": reused, "root_layout": layout,
            }
            already_matches = source_format == 29 if layout == "NZ" else source_format in (0, 2)
            if already_matches and not reused:
                raise ValueError(f"已匹配根布局的 {name} 没有复用 Native 存储")
        groups = {
            name: (fixture["metadata"][group["prefix"]], group["views"]) for name, group in fixture["groups"].items()
        }
        call = adapter.NativeCSACall(
            adapter.CSAOperators.register(),
            weights,
            fixture["hidden"],
            fixture["positions"],
            groups,
            layer_name=fixture["groups"]["compressed"]["prefix"],
            compact_metadata=fixture["compact"],
        )
        report["indexer_cache_binding"] = {
            "history_copy_before_csa": hasattr(call, "prepare_indexer_cache"),
            "native_storage_ptr": groups["indexer"][1][0].untyped_storage().data_ptr(),
            "root_views": {
                name: {"shape": list(value.shape), "stride": list(value.stride()),
                       "storage_offset": value.storage_offset(),
                       "storage_ptr": value.untyped_storage().data_ptr()}
                for name, value in call.args.items()
                if name in ("idx_kv_cache", "idx_kv_cache_shift64", "idx_native_kv_cache", "idx_kv_scale")
            },
        }
        restore(fixture)
        if args.save_case:
            ordered = {name: call.args[name] for name in module.decode_csa_tp1_layer_test.param_names}
            meta, payload = capture_tensors(
                ordered,
                argument_roles(root),
                layouts,
                {
                    "state_timing": "before_call",
                    "variant": args.variant,
                    "weight_nz_mode": args.weight_nz_mode,
                    "kind": "formal_layer_weights_synthetic_history",
                    "seed": args.seed,
                },
            )
            meta.update(layer_index=args.layer_index, tokens=fixture["tokens"])
            save_snapshot(args.output / "case", meta, payload)
        # 与生产路径 (vllm_ascend/models/pypto_deepseek_v4.py) 共用同一个开关，
        # 用来验证收窄 ring heap 之后 CSA 算子还能不能跑。
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs
        pypto.torch.init(
            device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
            **ring_sizing_kwargs(),  # 取证用：只认 PTO_CSA_RING_* 环境变量
            **({"enable_chip_swimlane": 4, "enable_dep_gen": True,
                "output_dir": str((args.output / "dfx").resolve())} if args.swimlane else {}),
        )
        pto = []
        report.update(pto_guards=[], root_layouts=layouts)
        for _ in range(2):
            restore(fixture)
            call()
            torch.npu.synchronize()
            pto.append(collect_state(fixture, call.args["x_out"], call.args["idx_topk"]))
            report["pto_guards"].append(guard_checks(fixture))
        report["pto_self"] = {name: compare_tensor(pto[1][name], value, 0, 0) for name, value in pto[0].items()}
        report["pto_native"] = {name: compare_tensor(pto[0][name], value, 0, 0) for name, value in native[0].items()}
        visible = (fixture["positions"].cpu() + 1) // 4
        report["topk_selection"] = compare_topk(pto[0]["idx_topk"], native[0]["idx_topk"], visible)
        if args.save_state:
            torch.save({"native": native[0], "pto": pto[0]}, args.output / "states.pt")
            report["saved_states"] = "states.pt"
        if args.save_case:
            torch.save({"state": native[0], "indexer_inputs": captured["indexer_inputs"]},
                       args.output / "native_reference.pt")
            torch.save({"idx_topk": pto[0]["idx_topk"], "idx_topk_scores": call.args["idx_topk_scores"].cpu()},
                       args.output / "pto_topk.pt")
            report["saved_reference"] = {
                "native": "native_reference.pt", "pto_topk": "pto_topk.pt",
                "scope": "逻辑 cache/state、层输出和 QLI 输入；不是单卡 bench 的根 ABI 参考",
            }
        report["status"] = "MEASURED"
        # 零容差只用于诊断差异；保护区破坏和非有限值仍是功能失败。
        for path in ("native_guards", "pto_guards"):
            if any(check["status"] != "PASS" for checks in report[path] for check in checks.values()):
                raise ValueError(f"{path}：Native slot 之外的存储被改写")
        for path in ("native_self", "pto_self", "pto_native"):
            if any(check.get("nonfinite") != 0 for check in report[path].values()):
                raise ValueError(f"{path}：输出/状态 shape、dtype 或有限值检查失败")
        if report["topk_selection"]["status"] == "FAIL":
            raise ValueError("Top-K 含越界、重复或缺失的候选索引")
        if args.swimlane:
            from dsv4_csa_single_card_bench import _export_swimlane

            # One root invocation includes all Native cache reads and writes.
            run_root = call

            def reset_swimlane():
                restore(fixture)

            replay = run_root
            if args.swimlane_graph:
                reset_swimlane()
                torch.npu.synchronize()
                swimlane_graph = torch.npu.NPUGraph()
                with torch.npu.graph(swimlane_graph):
                    run_root()
                replay = swimlane_graph.replay
                for _ in range(5):
                    reset_swimlane()
                    replay()
                torch.npu.synchronize()
            windows = []
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            for window in range(args.swimlane_windows):
                reset_swimlane()
                torch.npu.synchronize()
                pypto.torch.begin_dfx()
                try:
                    start.record()
                    replay()
                    end.record()
                finally:
                    pypto.torch.end_dfx()
                directory = args.output / "dfx"
                if window:
                    directory /= f"window_{window}"
                exported = _export_swimlane(directory)
                exported.update(
                    window=window, layer_index=args.layer_index, compact_metadata_policy="reuse",
                    input_source="formal_layer_weights_synthetic_history",
                    execution="graph_replay" if args.swimlane_graph else "eager",
                    profiled_event_us=start.elapsed_time(end) * 1000,
                    scope="单卡第二个 CSA 层；每窗口一次根调用；DFX 不替代无 profiler 性能计时",
                )
                windows.append(exported)
                report["swimlane_windows"] = windows
                if not exported["exported"]:
                    raise RuntimeError(f"泳道导出失败：{exported}")
            report["swimlane"] = windows[0]
        if args.graph:
            check_graph_replay(fixture, call, pto[0], report)
        if args.padding_graph:
            def make_call(compact):
                return adapter.NativeCSACall(
                    call.ops, weights, fixture["hidden"], fixture["positions"], groups,
                    layer_name=fixture["groups"]["compressed"]["prefix"], compact_metadata=compact,
                    buffers={name: call.args[name] for name in ("x_out", "idx_topk", "idx_topk_scores")},
                )

            check_padding_graph(fixture, call, pto[0], make_call, layer.self_attn.dsa_attn.dsa_attn.impl, report)
        if args.timing_iters:
            timing = {
                "status": "RUNNING", "iters": args.timing_iters, "warmup": args.timing_warmup,
                "scope": "单卡正式层权重、合成历史的 HC_pre→norm→CSA→HC_post 图重放设备区间；"
                         "含内部间隙，不是 16 卡 FULL_DECODE_ONLY 验收",
                "method": "图外 NPU Event 包住一次重放，含图派发可能留下的设备间隙；"
                          "每次在区间外恢复相同初态并毒化输出，无诊断拷贝；检查事件时间戳逐次更新",
                "order": ["native", "pto"],
                "compact_metadata_policy": args.timing_metadata,
                "pto_compact_metadata": (
                    "同一步第二个 CSA 层：复用首层已生成的两组 compact metadata，生成在区间外"
                    if args.timing_metadata == "reuse" else
                    "同一步首个 CSA 层：每次在图内生成两组 compact metadata"
                ),
                "native_compact_metadata": "Native 各层仍实际调用生产算子，计时保留原路径",
                "device": os.environ.get("TASK_DEVICE", str(args.device)),
            }
            report["timing"] = timing

            def timed_qli(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                # 只保留图内 Top-K 的返回引用；不把 clone/CPU 拷贝计入 Native 区间。
                captured["timed_topk"] = value[0]
                return value

            def native_call():
                with set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
                ):
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)

            def pto_call():
                # 对齐生产 service 的首层准备，不能提前生成 metadata 使 PTO 少计两个设备算子。
                impl = layer.self_attn.dsa_attn.dsa_attn.impl
                compact = {
                    name: impl._compute_compressor_metadata(groups[name][0].decode)
                    for name in ("compressed", "indexer")
                }
                prepared = adapter.NativeCSACall(
                    call.ops, weights, fixture["hidden"], fixture["positions"], groups,
                    layer_name=fixture["groups"]["compressed"]["prefix"], compact_metadata=compact,
                    buffers={name: call.args[name] for name in ("x_out", "idx_topk", "idx_topk_scores")},
                )
                prepared()

            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = timed_qli
            try:
                timing["native"] = measure_graph_interval(
                    fixture, native_call, output, lambda: captured["timed_topk"], native[0],
                    iters=args.timing_iters, warmup=args.timing_warmup,
                    require_exact=bool(args.deterministic_level),
                    profile_dir=args.output / "profile/native" if args.profile else None,
                )
            finally:
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
            timing["pto"] = measure_graph_interval(
                fixture, call if args.timing_metadata == "reuse" else pto_call,
                call.args["x_out"], lambda: call.args["idx_topk"], pto[0],
                iters=args.timing_iters, warmup=args.timing_warmup, require_exact=not ATOMIC_ADD,
                profile_dir=args.output / "profile/pto" if args.profile else None,
            )
            timing["native_over_pto_p50"] = timing["native"]["us_p50"] / timing["pto"]["us_p50"]
            timing["status"] = "MEASURED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--layer-index", type=int, default=2, choices=range(2, 43, 2),
                        help="正式 C4 层权重序号；4 对应模型第二个 CSA 层")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight-nz-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--variant", default="precision",
                        help="precision / performance，或 pkg:<包名> 指定实验包")
    parser.add_argument("--save-case", action="store_true")
    parser.add_argument("--save-state", action="store_true", help="保存两侧 8 类逻辑输出/状态，供跨布局逐元素比较")
    parser.add_argument("--save-sparse-case", action="store_true",
                        help="保存 Native 稀疏注意力的实际输入和逆 RoPE 前输出")
    parser.add_argument("--atomic-add", type=int, choices=(0, 1), help="0 为固定规约诊断；未指定时遵循环境配置")
    parser.add_argument("--graph", action="store_true", help="固定规约下验证同地址 A/B/A 输入的图重放")
    parser.add_argument("--padding-graph", action="store_true",
                        help="固定规约下，Native metadata 更新同一个图的满档/补位请求；batch 至少为 2")
    parser.add_argument("--deterministic-level", type=int, choices=(0, 1), default=1,
                        help="Native 确定性：默认 1 诊断，0 为性能部署；HCCL_DETERMINISTIC 同步设置")
    parser.add_argument("--timing-iters", type=int, default=0, help="两侧完整图重放设备区间采样次数；0 不计时")
    parser.add_argument("--timing-warmup", type=int, default=5, help="每侧计时图预热次数")
    parser.add_argument("--timing-metadata", choices=("reuse", "produce"), default="reuse",
                        help="默认 reuse 按同一步第二个 CSA 层复用 metadata；produce 单独测首层成本")
    parser.add_argument("--profile", action="store_true", help="计时后每侧单独采一次设备 profiler 核对区间与热点")
    parser.add_argument("--native-profile-only", action="store_true",
                        help="仅补采 Native 图重放及分算子 trace，不编译或执行 PTO")
    parser.add_argument("--swimlane", action="store_true",
                        help="单独采一次完整 PTO 根的 DFX 泳道，复用前层 metadata；在新的工作目录执行")
    parser.add_argument("--swimlane-windows", type=int, default=1, help="每个 DFX 窗口只执行一次根调用")
    parser.add_argument("--swimlane-graph", action="store_true", help="DFX 采 ACL Graph 重放；不采 eager 调用")
    args = parser.parse_args()
    if not 1 <= args.batch <= 40 or args.history < 0:
        parser.error("batch 必须为 1～40，history 不得为负")
    if args.padding_graph and args.batch < 2:
        parser.error("padding-graph 的 batch 至少为 2")
    if args.timing_iters < 0 or args.timing_warmup < 1:
        parser.error("timing-iters 不得为负，timing-warmup 至少为 1")
    if (args.profile or args.native_profile_only) and not args.timing_iters:
        parser.error("--profile/native-profile-only 须配合正数 --timing-iters")
    if args.native_profile_only and (args.profile or args.swimlane or args.graph or args.padding_graph
                                     or args.save_case or args.save_state):
        parser.error("--native-profile-only 不与 PTO 测量、正确性或状态保存选项混用")
    if args.swimlane and (args.timing_iters or args.profile or args.graph or args.padding_graph):
        parser.error("--swimlane 单独采集，不与图计时或图正确性窗口混用")
    if args.swimlane_windows < 1 or ((args.swimlane_graph or args.swimlane_windows != 1) and not args.swimlane):
        parser.error("泳道窗口数必须为正，--swimlane-graph/windows 须配合 --swimlane")
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.weight_nz_mode)
    os.environ["PTO_CSA_VARIANT"] = args.variant
    if args.atomic_add is not None:
        os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = str(args.atomic_add)
    os.environ["HCCL_DETERMINISTIC"] = "true" if args.deterministic_level else "false"
    report = {
        "status": "RUNNING",
        "batch": args.batch,
        "history": args.history,
        "layer_index": args.layer_index,
        "seed": args.seed,
        "weight_nz_mode": args.weight_nz_mode,
        "variant": args.variant,
        "native_profile_only": args.native_profile_only,
        "scope": "正式单层权重、合成输入/历史；零容差差异诊断，不代表数值或整模型验收",
        "deterministic_level": args.deterministic_level,
        "hccl_deterministic": bool(args.deterministic_level),
        "checkpoint": str(args.checkpoint),
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        if report.get("timing", {}).get("status") == "RUNNING":
            report["timing"]["status"] = "FAIL"
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("weights", "layouts")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
