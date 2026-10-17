# SPDX-License-Identifier: Apache-2.0
"""用真实Native半层核对S1/S5动态CSA；可另采原分task真实S6因果前缀。

复用正式权重、Native builder、页布局和保护区；直接绑定Native device输入。
Dspark上限保持5，仅本轮实际query长度为命令行值。Native只作功能/精度参照，
不采其性能，也不替代七档S6的原动态PTO性能基线。零容差报告如实保留差异；
shape/有限值/写保护与eager/graph稳定性和Native精度分别记账，不混称完整通过。
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


def bind_native_inputs(adapter, layer, fixture, output, torch):
    """机械绑定已有Native存储；不修改旧S6适配入口或构造伪S6请求。

    PreparedDynamicCSA沿用旧参数命名，但不需要旧NativeCSACall的S6闸门。
    此处只收集现有Tensor与view；逐请求几何继续由正式算子从Q/L读取。
    """
    groups = fixture["groups"]
    req = {name: fixture["metadata"][g["prefix"]].decode for name, g in groups.items()}
    indexer_metadata = fixture["metadata"][groups["indexer"]["prefix"]]
    root = adapter.prepare_weights(layer.self_attn, indexer_metadata.hadamard, layer)
    root.update(
        x_hc=fixture["hidden"], x_out=output, position_ids=fixture["positions"],
        kv_cache=groups["swa"]["views"][0], cmp_kv=groups["compressed"]["views"][0],
        idx_native_kv_cache=adapter.indexer_storage(*groups["indexer"]["views"]),
        kv_seq_lens=req["indexer"].seq_lens,
        cmp_query_start_loc=req["compressed"].query_start_loc,
        cmp_seq_lens=req["compressed"].seq_lens,
        idx_query_start_loc=req["indexer"].query_start_loc,
        compress_state=adapter.physical_pages(groups["state"]["views"][0]),
        inner_compress_state=adapter.physical_pages(groups["indexer_state"]["views"][0]),
        idx_topk=torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=output.device),
        idx_topk_scores=torch.empty((fixture["tokens"], 512), dtype=torch.float32, device=output.device),
    )
    for name, key in (("swa", "ori"), ("compressed", "cmp"), ("indexer", "idx"),
                      ("state", "state"), ("indexer_state", "inner_state")):
        root[key + "_block_table"] = adapter.table_storage(req[name].block_table)
    for name, key in (("swa", "ori"), ("state", "state"), ("indexer_state", "inner_state")):
        root[key + "_slot_mapping"] = req[name].slot_mapping
    for name, prefix in (("compressed", "cmp"), ("indexer", "inner")):
        cos, sin, slots = fixture["compact"][name]
        root[prefix + "_freqs_cos"] = cos.view(-1, 64)
        root[prefix + "_freqs_sin"] = sin.view(-1, 64)
        root[("cmp" if name == "compressed" else "idx") + "_slot_mapping"] = slots
    layer_name = layer.self_attn.dsa_attn.dsa_attn.layer_name
    for kind in ("cos", "sin"):
        root["freqs_" + kind] = getattr(req["compressed"], kind)[layer_name][:fixture["tokens"]].view(-1, 64)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--query-tokens", type=int, choices=(1, 5, 6), default=5)
    parser.add_argument("--causal-extension-reference", action="store_true")
    parser.add_argument("--spmd-source", type=Path)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须通过task-submit --device auto提交")
    if args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("设备参数必须与队列分配一致")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage="initializing", passed=False, functional_passed=False, precision_exact=False,
                  device_execution=False, performance_measured=False, batch=args.batch,
                  history=args.history, query_tokens=args.query_tokens, dspark_spec_tokens=5,
                  reference="真实Native attention半层；原分task仍有S6假设，不用于此S5比较",
                  limitations=["不是S5性能验收", "不是mixed-query或连续状态轨迹", "服务路由未切换"],
                  rounds=[])

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "passed", "device_execution")}), flush=True)

    try:
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor, compare_topk
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.utils import enable_custom_op

        current_platform.pre_register_and_update()
        device_id = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        if not enable_custom_op():
            raise RuntimeError("Native自定义算子包尚未注册")
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, args.history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        import pypto.torch
        from pypto import passes
        from pypto.runtime import CompileOptions
        from compile_diagnostics import compilation_probe
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
        if args.spmd_source:
            # 只在本测试进程导入明确版本的正式动态源码，不切换共享工作区。
            package_name = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd"
            if any(name == package_name or name.startswith(package_name + ".") for name in sys.modules):
                raise RuntimeError("SPMD包已导入，不能在进程中途换源码")
            source = args.spmd_source.resolve()
            spec = importlib.util.spec_from_file_location(
                package_name, source / "__init__.py", submodule_search_locations=[str(source)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
            report["isolated_spmd_source"] = str(source)
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.runtime_abi import PHASE_COUNT
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        package = Path(importlib.import_module(PreparedDynamicCSA.__module__).__file__).parent
        (args.output / "source").mkdir()
        for path in package.glob("*.py"):
            (args.output / "source" / path.name).write_text(path.read_text())
        (args.output / Path(__file__).name).write_text(Path(__file__).read_text())
        device = torch.device(f"npu:{device_id}")
        with native_session(config, device_id), torch.inference_mode():
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device,
                                   query_tokens=args.query_tokens)
            report["stage"] = "fixture_ready"
            save()
            scale = fixture["groups"]["indexer"]["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            fixture["groups"]["indexer"]["initial"] = fixture["groups"]["indexer"]["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            root = bind_native_inputs(adapter, layer, fixture, output, torch)
            allocations = {}

            def allocate(name, shape, dtype, device):
                count = 1
                for n in shape:
                    count *= n
                guard = 512 // torch.empty((), dtype=dtype).element_size()
                storage = torch.full((count + guard * 2,), 37, dtype=dtype, device=device)
                allocations[name] = storage, guard
                return storage[guard:-guard].view(shape)

            candidate = PreparedDynamicCSA(root, max_query_tokens=args.query_tokens,
                                           max_seq_len=args.history + args.query_tokens, allocator=allocate)
            report["policy"] = candidate.policy.constexprs()
            report["tensor_count"] = len(candidate.buffers)
            pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             **ring_sizing_kwargs())
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            captured = {}
            original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

            def capture_topk(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                captured["topk"] = value[0].detach().clone()
                return value

            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = capture_topk
            try:
                restore(fixture)
                with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                               num_actual_tokens=fixture["tokens"]):
                    _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)
                torch.npu.synchronize()
                reference = collect_state(fixture, output, captured["topk"])
                report["native_guards"] = guard_checks(fixture)
                assert all(v["status"] == "PASS" for v in report["native_guards"].values())
            finally:
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
            report["stage"] = "native_reference_ready"
            save()
            restore(fixture)
            for name, value in candidate.allocations.items():
                if name not in ("signal", "ffts"):
                    value.fill_(37)
            with compilation_probe(args.output / "compile_diagnostics", wait_backoff=True):
                candidate(config=options)
            torch.npu.synchronize()
            report["device_execution"] = True
            eager = collect_state(fixture, output, root["idx_topk"])
            report["native_comparison"] = {k: compare_tensor(v, reference[k], 0, 0) for k, v in eager.items()}
            report["native_topk_selection"] = compare_topk(
                eager["idx_topk"], reference["idx_topk"], (fixture["positions"].cpu() + 1) // 4)
            report["eager_guards"] = guard_checks(fixture)
            addresses = {k: v.data_ptr() for k, v in candidate.buffers.items()}
            restore(fixture)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                candidate(config=options)
            for generation in range(2):
                restore(fixture)
                for name, value in candidate.allocations.items():
                    if name not in ("signal", "ffts"):
                        value.fill_(37)
                before = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0]
                graph.replay()
                torch.npu.synchronize()
                actual = collect_state(fixture, output, root["idx_topk"])
                checks = {k: compare_tensor(v, eager[k], 0, 0) for k, v in actual.items()}
                guards = guard_checks(fixture)
                scratch = {k: bool((v[:g].cpu() == 37).all() and (v[-g:].cpu() == 37).all())
                           for k, (v, g) in allocations.items()}
                signal = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
                protocol = bool((signal[0, :, 0] == before + 1).all() and (signal[:, :, 1:] == 0).all())
                same = addresses == {k: v.data_ptr() for k, v in candidate.buffers.items()}
                passed = protocol and same and all(scratch.values()) and all(
                    v["status"] == "PASS" for v in (*checks.values(), *guards.values()))
                report["rounds"].append(dict(replay=generation + 1, passed=passed, protocol=protocol,
                                             same_addresses=same, comparisons=checks, guards=guards, scratch=scratch))
                assert passed, "实际query的eager/graph或保护区/代际失败"
            report["functional_passed"] = all(v["status"] == "PASS" for v in report["eager_guards"].values()) and all(
                v.get("nonfinite") == 0 for v in report["native_comparison"].values()) and report["native_topk_selection"]["status"] != "FAIL"
            report["precision_exact"] = all(v["status"] == "PASS" for v in report["native_comparison"].values())
            report["passed"] = report["functional_passed"] and report["precision_exact"]
            # 先保存真实SQ的数值证据；后续因果参照失败也不丢失已完成的比较。
            if not report["precision_exact"]:
                torch.save({"native_output": reference["x_out"], "spmd_output": eager["x_out"],
                            "native_topk": reference["idx_topk"], "spmd_topk": eager["idx_topk"]},
                           args.output / "numerical_diagnostic.pt")
            if args.causal_extension_reference:
                from causal_extension import validate as validate_causal_extension

                (args.output / "causal_extension.py").write_text(
                    Path(__file__).with_name("causal_extension.py").read_text())
                report["stage"] = "causal_extension_reference"
                save()
                report["causal_extension"] = validate_causal_extension(
                    adapter, layer, config, fixture, eager, root["idx_topk_scores"].cpu(),
                    options, args.output / "causal_extension.json")
            report["stage"] = "complete"
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
