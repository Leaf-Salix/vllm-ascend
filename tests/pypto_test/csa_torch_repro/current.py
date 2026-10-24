# SPDX-License-Identifier: Apache-2.0
"""当前内部工作区 TND-SPMD/MPMD 的纯 PyTorch 单卡验收。"""

import argparse
import importlib
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

from csa_mpmd_tnd_20261021.run_case import check_case, export_l1, write_json
from dsv4_csa_validation import compare_tensor
from inputs import make_inputs, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="C04")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-windows", type=int, default=0, help="0仅验证功能；采L1时至少10个窗口/实现")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    if args.profile_windows and args.profile_windows < 10:
        parser.error("正式L1采样每种实现至少10个窗口")
    manifest = json.loads(args.cases.read_text())
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    check_case(case)
    args.output.mkdir(parents=True, exist_ok=False)
    toolchain = json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text())
    report = dict(
        stage="initialize",
        passed=False,
        case=case,
        toolchain=toolchain,
        scope="纯PyTorch单卡算子功能；不代表模型图重放或Native精度一致",
        ring_config=dict(heap_mb=[16, 256, 16, 16], task_window=4096),
        checks={},
        superkernel=False,
        npugraph_ex_static_kernel=False,
    )

    def save(stage):
        report["stage"] = stage
        write_json(args.output / "result.json", report)
        print("TORCH_ONLY_CURRENT " + stage, flush=True)

    try:
        import pypto.torch
        import torch
        import torch_npu
        from pypto import passes
        from pypto.runtime import CompileOptions

        packages = {
            "spmd": "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd",
            "mpmd": "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd",
        }
        if args.cpu:
            for side, package in packages.items():
                module = importlib.import_module(package + ".entry")
                name = "decode_csa_spmd" if side == "spmd" else "decode_csa_tnd"
                for suffix in ("", "_debug"):
                    getattr(module, name + suffix)._get_dep_graph()
            report["cpu_sources_parsed"] = True
            save("cpu_sources_parsed")
            return
        assigned = os.environ.get("TASK_DEVICE", "")
        if not assigned.isdigit():
            raise ValueError("须由设备队列分配一张卡并设置TASK_DEVICE")
        device_id = int(assigned)
        if args.device is not None and args.device != device_id:
            raise ValueError("--device与队列分配不一致")
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(1)
        device = torch.device(f"npu:{device_id}")
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        pypto.torch.init(
            device=device_id,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            ring_heap=[n * 1024 * 1024 for n in report["ring_config"]["heap_mb"]],
            ring_task_window=[4096] * 4,
            enable_chip_swimlane=int(bool(args.profile_windows)),
            enable_dep_gen=False,
            output_dir=str(args.output / "dfx"),
        )
        with torch.inference_mode():
            native, groups, readonly, report["weights"] = make_inputs(case, args.model, manifest["layer"], device)
            normal_inputs = {k: v for k, v in native.items() if k not in ("idx_topk", "idx_topk_scores")}
            entries, diagnostics = {}, {}
            for side, package in packages.items():
                binding = importlib.import_module(package + ".binding").PreparedDynamicCSA
                kwargs = dict(max_query_tokens=6, max_seq_len=max(case["seq_lens"]))
                entries[side] = binding(normal_inputs, **kwargs)
                diagnostics[side] = binding(native, **kwargs)
                assert not entries[side].allocations and not diagnostics[side].allocations
                assert "topk_indices" not in entries[side].buffers
                assert len(entries[side].buffers) == 55
            report["external_workspace_bytes"] = {side: 0 for side in entries}
            report["normal_tensor_arguments"] = {side: len(entry.buffers) for side, entry in entries.items()}
            save("inputs_ready")

            def snapshot(debug):
                values = {"output": native["x_out"][: case["valid_tokens"]].cpu().clone()}
                if debug:
                    values.update(
                        topk=native["idx_topk"][: case["valid_tokens"]].cpu().clone(),
                        scores=native["idx_topk_scores"][: case["valid_tokens"]].cpu().clone(),
                    )
                guards = {}
                for name, group in groups.items():
                    raw = group["allocation"].cpu()
                    guards[name] = bool(((raw == group["initial"]) | group["allowed"]).all())
                    pairs = sorted({(p, r) for p, r in group["slots"].tolist() if p >= 0 and r >= 0})
                    pages, rows = [p for p, _ in pairs], [r for _, r in pairs]
                    for i, view in enumerate(group["views"]):
                        current = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                        values[f"{name}.{i}"] = current[pages, rows].clone()
                for name, (value, initial) in readonly.items():
                    guards["readonly/" + name] = torch.equal(value.cpu(), initial)
                if not debug:
                    guards["diagnostic_topk_untouched"] = bool((native["idx_topk"] == -1).all())
                    guards["diagnostic_scores_untouched"] = bool((native["idx_topk_scores"] == 0).all())
                return values, guards

            def check(side, stage, expected, debug=False):
                actual, guards = snapshot(debug)
                if expected is None:
                    expected = actual
                numerical = {}
                for name, value in actual.items():
                    rule = manifest["correctness"][
                        "floating_output"
                        if name == "output"
                        else "topk_scores"
                        if name == "scores"
                        else "floating_cache_state"
                    ]
                    atol, rtol = (rule["atol"], rule["rtol"]) if value.is_floating_point() else (0, 0)
                    if value.numel():
                        numerical[name] = compare_tensor(value, expected[name], atol, rtol)
                        numerical[name]["bit_equal"] = torch.equal(
                            value.view(torch.uint8), expected[name].view(torch.uint8)
                        )
                    else:
                        numerical[name] = dict(status="PASS", elements=0, bit_equal=True)
                report["checks"][side + "_" + stage] = dict(guards=guards, numerical=numerical)
                assert all(guards.values()) and all(r["status"] == "PASS" for r in numerical.values()), (side, stage)
                return expected

            expected, tables = None, {}
            build = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
            for side, entry in diagnostics.items():
                save("diagnostic_eager_" + side)
                restore(native, groups)
                entry(config=options)
                torch.npu.synchronize()
                expected = check(side, "diagnostic_eager", expected, debug=True)
            for side, entry in entries.items():
                save("normal_eager_" + side)
                restore(native, groups)
                before = set(build.glob("*/kernel_config.py"))
                entry(config=options)
                torch.npu.synchronize()
                created = set(build.glob("*/kernel_config.py")) - before
                if args.profile_windows:
                    assert len(created) == 1, (side, created)
                    tables[side] = created.pop()
                check(side, "normal_eager", expected)
            # 两档图和对应入口一起保留；不能提前释放仍被共享图池引用的对象。
            graphs = {}
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for side, entry in entries.items():
                    save("capture_" + side)
                    restore(native, groups)
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        entry(config=options)
                    graphs[side] = graph
                    for repeat in range(2):
                        restore(native, groups)
                        graph.replay()
                        stream.synchronize()
                        check(side, f"graph_{repeat}", expected)
                if args.profile_windows:
                    for _ in range(3):
                        for graph in graphs.values():
                            restore(native, groups)
                            graph.replay()
                            stream.synchronize()
                    records = []
                    save("paired_l1")
                    for sample in range(args.profile_windows):
                        order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
                        for side in order:
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
                            records.append(
                                export_l1(
                                    source,
                                    args.output / f"l1_{sample:02d}_{side}",
                                    side,
                                    tables[side],
                                    allow_spmd_prologue=True,
                                )
                            )
                    report["windows"] = records
                    report["statistics_us"] = {}
                    for side in entries:
                        values = [r["incore_span_us"] for r in records if r["side"] == side]
                        report["statistics_us"][side] = dict(
                            min=min(values),
                            median=statistics.median(values),
                            mean=statistics.mean(values),
                            max=max(values),
                            samples=values,
                        )
                        restore(native, groups)
                        graphs[side].replay()
                        stream.synchronize()
                        check(side, "post_l1", expected)
            torch.npu.synchronize()
            assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)
            assert not any(name.startswith("_C_ascend::") for name in torch._C._dispatch_get_all_op_names())
            report["dependency_check"] = dict(vllm_imported=False, native_custom_ops_registered=False)
            report["passed"] = True
            save("complete")
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        save("failed")
        raise


if __name__ == "__main__":
    main()
