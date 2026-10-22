# SPDX-License-Identifier: Apache-2.0
"""固定 TND 场景的独立 PyTorch 单卡对照，无 vLLM/Native 自定义算子依赖。"""

import argparse
import importlib
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

from csa_mpmd_tnd_20261021.run_case import check_case, export_l1, mount_package, write_json
from dsv4_csa_validation import compare_tensor
from inputs import make_inputs, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="h131072_b16")
    parser.add_argument("--cases", type=Path, default=Path(__file__).parents[1] / "csa_mpmd_tnd_20261021/cases.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control-mpmd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.cases.read_text())
    case = next(item for item in manifest["cases"] if item["name"] == args.case)
    check_case(case)
    report = {"stage": "initializing", "passed": False, "case": case,
              "scope": "PyTorch-only single-card, full CSA L1", "reference_commit": manifest["reference_commit"],
              "superkernel": False, "npugraph_ex_static_kernel": False, "checks": {}}

    def save(stage):
        report["stage"] = stage
        write_json(args.output / "result.json", report)
        print(f"TORCH_ONLY_STAGE {stage}", flush=True)

    try:
        import torch
        import torch_npu
        import pypto.torch
        from pypto import passes
        from pypto.runtime import CompileOptions

        root = args.frozen / "vllm_ascend/ops/pypto"
        prefix = "vllm_ascend.ops.pypto."
        mount_package(root / "deepseek_v4_flash_csa", prefix + "deepseek_v4_flash_csa")
        packages = {"spmd": mount_package(root / "deepseek_v4_flash_csa_spmd_tnd", prefix + "_torch_reference")}
        if args.control_mpmd:
            packages["mpmd_control"] = mount_package(args.control_mpmd, prefix + "_torch_control")
        packages["mpmd"] = mount_package(args.candidate, prefix + "_torch_candidate")
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        if args.cpu:
            for side, package in packages.items():
                entry = importlib.import_module(package + ".entry")
                function = entry.decode_csa_spmd if side == "spmd" else entry.decode_csa_tnd
                function._get_dep_graph()
            report["passed"] = True
            save("cpu_sources_parsed")
            return
        device_id = int(os.environ["TASK_DEVICE"])
        if args.device is not None and args.device != device_id:
            raise ValueError("设备参数与队列分配不符")
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device(f"npu:{device_id}")
        with torch.inference_mode():
            native, groups, readonly, report["weights"] = make_inputs(case, args.model, manifest["layer"], device)
            entries, guards = {}, {}
            for side, package in packages.items():
                def allocate(name, shape, dtype, device, side=side):
                    count = 1
                    for dim in shape:
                        count *= dim
                    pad = 512 // torch.empty((), dtype=dtype).element_size()
                    allocation = torch.full((count + 2 * pad,), 37, dtype=dtype, device=device)
                    guards[side, name] = allocation, pad
                    return allocation[pad:-pad].view(shape)

                entries[side] = importlib.import_module(package + ".binding").PreparedDynamicCSA(
                    native, max_query_tokens=6, max_seq_len=max(case["seq_lens"]), allocator=allocate)
            report["descriptors"] = {side: {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                            for name, value in entry.buffers.items()} for side, entry in entries.items()}
            pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             enable_chip_swimlane=1, enable_dep_gen=False, output_dir=str(args.output / "dfx"))
            save("inputs_ready")

            def snapshot(side):
                values = {"output": native["x_out"].cpu(), "topk": native["idx_topk"].cpu(),
                          "scores": native["idx_topk_scores"].cpu()}
                checks = {}
                for name, group in groups.items():
                    raw = group["allocation"].cpu()
                    checks[name] = bool(((raw == group["initial"]) | group["allowed"]).all())
                    pairs = sorted({(p, r) for p, r in group["slots"].tolist() if p >= 0 and r >= 0})
                    pages, rows = [p for p, _ in pairs], [r for _, r in pairs]
                    for i, view in enumerate(group["views"]):
                        current = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                        values[f"{name}.{i}"] = current[pages, rows].clone()
                for name, (value, initial) in readonly.items():
                    checks["readonly/" + name] = torch.equal(value.cpu(), initial)
                for (owner, name), (allocation, pad) in guards.items():
                    if owner == side:
                        checks["scratch/" + name] = bool((allocation[:pad].cpu() == 37).all()
                                                          and (allocation[-pad:].cpu() == 37).all())
                return values, checks

            def check(side, stage, expected):
                actual, guards_ok = snapshot(side)
                if expected is None:
                    expected = actual
                numerical = {}
                for name, value in actual.items():
                    rule = manifest["correctness"]["floating_output" if name == "output" else
                                                   "topk_scores" if name == "scores" else "floating_cache_state"]
                    atol, rtol = (rule["atol"], rule["rtol"]) if value.is_floating_point() else (0, 0)
                    if value.numel():
                        numerical[name] = compare_tensor(value, expected[name], atol, rtol)
                        numerical[name]["bit_equal"] = torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8))
                    else:
                        numerical[name] = {"status": "PASS", "elements": 0, "bit_equal": True}
                report["checks"][side + "_" + stage] = {"guards": guards_ok, "numerical": numerical}
                if not all(guards_ok.values()) or not all(item["status"] == "PASS" for item in numerical.values()):
                    raise AssertionError(f"{side}/{stage}: 数值或保护区失败")
                return expected

            build = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
            expected, tables = None, {}
            for side, entry in entries.items():
                save("compile_and_eager_" + side)
                restore(native, groups)
                before = set(build.glob("*/kernel_config.py"))
                entry(config=options)
                torch.npu.synchronize()
                created = set(build.glob("*/kernel_config.py")) - before
                if len(created) != 1:
                    raise RuntimeError(f"{side}: 无法定位唯一根kernel配置")
                tables[side] = created.pop()
                expected = check(side, "eager", expected)
                save("eager_passed_" + side)
            graphs = {}
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for side, entry in entries.items():
                    restore(native, groups)
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        entry(config=options)
                    graphs[side] = graph
                    restore(native, groups)
                    graph.replay()
                    stream.synchronize()
                    check(side, "graph", expected)
                for sample in range(3):
                    for side in list(graphs) if sample % 2 == 0 else list(reversed(graphs)):
                        restore(native, groups)
                        graphs[side].replay()
                        stream.synchronize()
                records = []
                save("paired_l1")
                for sample in range(5):
                    for side in list(graphs) if sample % 2 == 0 else list(reversed(graphs)):
                        restore(native, groups)
                        stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        source = args.output / "dfx"
                        if records:
                            source /= f"window_{len(records)}"
                        records.append(export_l1(source, args.output / f"l1_{sample:02d}_{side}", side, tables[side]))
                for side, graph in graphs.items():
                    restore(native, groups)
                    graph.replay()
                    stream.synchronize()
                    check(side, "post_l1", expected)
            report["windows"] = records
            report["statistics_us"] = {}
            for side in entries:
                values = [item["incore_span_us"] for item in records if item["side"] == side]
                report["statistics_us"][side] = {"min": min(values), "mean": statistics.mean(values),
                                                 "max": max(values), "samples": values}
            if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
                raise AssertionError("独立入口意外加载了 vLLM")
            if any(name.startswith("_C_ascend::") for name in torch._C._dispatch_get_all_op_names()):
                raise AssertionError("独立入口意外注册了 Native 自定义算子")
            report["dependency_check"] = {"vllm_imported": False, "native_custom_ops_registered": False}
            report["passed"] = True
            save("complete")
    except BaseException as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save("failed")
        raise


if __name__ == "__main__":
    main()
