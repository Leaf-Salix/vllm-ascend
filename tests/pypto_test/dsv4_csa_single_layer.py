# SPDX-License-Identifier: Apache-2.0
"""当前 release 的单卡 Native/PTO 整层诊断；不加载 MoE，不代替整模型验收。"""

import argparse
import importlib
import json
import os
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


def make_layer(config, checkpoint, device):
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
                prefix="model.layers.2.self_attn",
            )
            self.input_layernorm = RMSNorm(hf.hidden_size, eps=hf.rms_norm_eps)

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            layer = AttentionHalf()
    finally:
        torch.set_default_dtype(old_dtype)
    records, methods = load_formal_layer_weights(layer.self_attn, checkpoint)
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm.weight"):
        key = f"layers.2.{name}"
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


def run(args, report):
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
    torch_npu.npu.set_deterministic_level(1)
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
        layer, details = make_layer(config, args.checkpoint, device)
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
        captured = {}

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
        report["native_self"] = {
            name: compare_tensor(native[1][name], value, 0, 0) for name, value in native[0].items()
        }

        import pypto.torch

        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.nz_mode import root_weight_layouts
        from vllm_ascend.ops.pypto.variant import variant_package

        package = variant_package()
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import ATOMIC_ADD

        reduction = importlib.import_module(f"{package}.qkv_proj_rope")
        report["pto_reduction"] = {
            "atomic_add": ATOMIC_ADD, "qr_split_k": reduction.QR_OK, "kv_split_k": reduction.KV_OK,
        }
        if args.graph and ATOMIC_ADD:
            raise ValueError("--graph 使用逐元素精确比较，须设置 --atomic-add 0 排除跨核规约波动")
        adapter = importlib.import_module(f"{package}.native_adapter")
        module = importlib.import_module(f"{package}.decode_csa")
        root = module._decode_csa_tp1_layer
        layouts = root_weight_layouts(root)
        hadamard = fixture["metadata"][fixture["groups"]["indexer"]["prefix"]].hadamard
        weights = adapter.prepare_weights(layer.self_attn, hadamard, layer)
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
            meta.update(layer_index=2, tokens=fixture["tokens"])
            save_snapshot(args.output / "case", meta, payload)
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
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
        if args.graph:
            check_graph_replay(fixture, call, pto[0], report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight-nz-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--variant", choices=("precision", "performance"), default="precision")
    parser.add_argument("--save-case", action="store_true")
    parser.add_argument("--atomic-add", type=int, choices=(0, 1), help="0 为固定规约诊断；未指定时遵循环境配置")
    parser.add_argument("--graph", action="store_true", help="固定规约下验证同地址 A/B/A 输入的图重放")
    args = parser.parse_args()
    if not 1 <= args.batch <= 40 or args.history < 0:
        parser.error("batch 必须为 1～40，history 不得为负")
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.weight_nz_mode)
    os.environ["PTO_CSA_VARIANT"] = args.variant
    if args.atomic_add is not None:
        os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = str(args.atomic_add)
    os.environ["HCCL_DETERMINISTIC"] = "true"
    report = {
        "status": "RUNNING",
        "batch": args.batch,
        "history": args.history,
        "seed": args.seed,
        "weight_nz_mode": args.weight_nz_mode,
        "variant": args.variant,
        "scope": "正式单层权重、合成输入/历史；零容差差异诊断，不代表数值或整模型验收",
        "deterministic_level": 1,
        "hccl_deterministic": True,
        "checkpoint": str(args.checkpoint),
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
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
