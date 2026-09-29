"""Read the completed Native named-backend A/B; never start a device job."""

import collections
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 16), (8192, 24))
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
          "indexer.0", "indexer.1", "indexer_state.0"}
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same task: {status}")
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location(
        "native_super_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    pair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    source = read(ROOT / "source.json")
    result = {"task": task, "task_status": status, "source": source, "cases": [],
              "scope": "Native superkernel on/off; direct named backend; not PTO or EP16 acceptance"}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}"
        sides = {}
        reports = {}
        for flag in (0, 1):
            path = folder / f"super{flag}"
            report = read(path / "report.json")
            reports[flag] = report
            if (report["source"], report["history"], report["batch"], report["side"]) != (
                    source["source"], history, batch, "native"):
                raise ValueError("Wrong source, shape or backend side")
            if report["compile_entry"] != "torch.compile(backend=npugraph_ex)":
                raise ValueError("Native did not use the requested named backend")
            options = report["backend_options"]
            if options["force_eager"] or not options["static_kernel_compile"]:
                raise ValueError("Capture owner or static-kernel configuration differs")
            if bool(report["super_kernel"]) != bool(flag):
                raise ValueError("Wrong requested superkernel flag")
            if not report["compiler"]["static_super_flags"] or any(
                    value != bool(flag) for value in report["compiler"]["static_super_flags"]):
                raise ValueError("Static compiler did not receive the requested flag")
            calls = report["super_kernel_graph_calls"]
            if bool(calls) != bool(flag) or any(call["status"] != "RETURNED_SUCCESSFULLY" for call in calls):
                raise ValueError("Backend graph-optimization evidence is incomplete")
            if not report["multistream"]["dsa_overlap"]:
                raise ValueError("Native multistream DSA was disabled")
            metrics = pair.analyze_side(path)
            with Path(metrics["profile_csv"]).open() as stream:
                kernels = list(csv.DictReader(stream))
            streams = collections.Counter(row["Stream ID"] for row in kernels)
            if not flag and len(streams) < 2:
                raise ValueError("Baseline profile does not demonstrate Native multistream execution")
            metrics.update(samples_us=report["timing"]["samples_us"], stream_kernel_counts=dict(streams),
                           backend_options=options, super_kernel_graph_calls=calls,
                           superkernel_profile_rows=[row for row in kernels if "super" in
                                                     (row["Name"] + row["Type"]).lower()])
            if flag and not metrics["superkernel_profile_rows"]:
                raise ValueError("No SuperKernel execution observed in the enabled profile")
            metrics["profile_device_span_us"] = (
                max(float(row["Start Time(us)"]) + float(row["Duration(us)"]) for row in kernels)
                - min(float(row["Start Time(us)"]) for row in kernels))
            sides[f"super{flag}"] = metrics
        for field in ("device", "cann", "requested", "multistream", "compilation_scope"):
            if reports[0][field] != reports[1][field]:
                raise ValueError(f"A/B differs beyond the intended switch: {field}")
        off_options = dict(reports[0]["backend_options"])
        on_options = dict(reports[1]["backend_options"])
        off_options.pop("super_kernel_optimize")
        on_options.pop("super_kernel_optimize")
        if off_options != on_options:
            raise ValueError("Other compiler options changed")
        states = {flag: torch.load(folder / f"super{flag}/states.pt", map_location="cpu", weights_only=True)
                  for flag in (0, 1)}
        if any(set(values) != STATES for values in states.values()):
            raise ValueError("Incomplete output/cache/state coverage")
        checks = {name: compare_tensor(states[1][name], states[0][name], 0, 0) for name in sorted(STATES)}
        del states
        before, after = sides["super0"]["mean_us"], sides["super1"]["mean_us"]
        result["cases"].append({"history": history, "batch": batch, "sides": sides,
                                "state_checks": checks, "change_pct": 100 * (after / before - 1)})
    result["weighted_8_2_change_pct"] = sum(
        weight * case["change_pct"] for weight, case in zip((.8, .2), result["cases"]))
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Native superkernel：显式npugraph_ex入口", "",
             "CANN9.2，static kernel均开启；单位μs，5预热/20次编译调用的图外事件计时，profile另采。", "",
             "| 档位 | 关闭 | 开启 | 耗时变化 | 关闭/开启P95 | 关闭/开启max | 状态零容差 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for case in result["cases"]:
        off, on = (case["sides"][key] for key in ("super0", "super1"))
        checks = collections.Counter(v["status"] for v in case["state_checks"].values())
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {off['mean_us']:.3f} | {on['mean_us']:.3f} "
                     f"| {case['change_pct']:+.3f}% | {off['us_p95']:.3f}/{on['us_p95']:.3f} "
                     f"| {off['us_max']:.3f}/{on['us_max']:.3f} | {dict(checks)} |")
    lines += ["", f"长短8:2耗时变化：{result['weighted_8_2_change_pct']:+.3f}%。", "",
              "静态编译开关、实际图优化调用、profile kernel与stream、状态差异均见evidence.json。",
              "图优化API成功不等于所有Native自定义kernel均已融合；零容差差异须结合det0归约规则判断。",
              "编译调用事件区间可能含主机参数处理/派发空闲，不与旧直接图重放均值拼表。",
              "依用户要求，无明确收益即结束此方向；不自动以微小均值变化宣布采用。",
              "本报告不是原vLLM包装基线的新标签，也不是PTO或整模型验收。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
