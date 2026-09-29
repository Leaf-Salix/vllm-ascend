"""Collect terminal fixed-address replay evidence without starting device work."""

import collections
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
          "indexer.0", "indexer.1", "indexer_state.0"}


def read(path):
    return json.loads(path.read_text())


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same device task: {status}")
    spec = importlib.util.spec_from_file_location(
        "native_graph_replay_metrics", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    pair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    source = read(ROOT / "source.json")
    result = {"task": task, "task_status": status, "source": source, "cases": [],
              "scope": "Fixed-address backend-created graph; no new superkernel A/B or EP16 acceptance"}
    for history, batch in source["cases"]:
        folder = ROOT / f"h{history}_b{batch}/super1"
        report = read(folder / "report.json")
        if (report["source"], report["history"], report["batch"], report["side"]) != (
                source["source"], history, batch, "native"):
            raise ValueError("Unexpected source, shape or backend side")
        options = report["backend_options"]
        if (report["compile_entry"] != "torch.compile(backend=npugraph_ex)"
                or options["force_eager"] or not options["static_kernel_compile"]
                or not options["super_kernel_optimize"] or not report["super_kernel"]):
            raise ValueError("Native backend or adopted compilation options changed")
        flags = report["compiler"]["static_super_flags"]
        calls = report["super_kernel_graph_calls"]
        if not flags or not all(flags) or not calls or any(
                call["status"] != "RETURNED_SUCCESSFULLY" for call in calls):
            raise ValueError("Superkernel was not applied by the backend")
        measurement = report["measurement"]
        if (measurement["graph_count"] != 1 or measurement["host_updated_nodes"] != 0
                or measurement["timed_call"] != "backend-created NPUGraph.replay"):
            raise ValueError("Direct replay assumptions are not satisfied")
        checks = report["timing"]["eager_comparison"]
        if set(checks) != STATES or any(check["status"] != "PASS" for check in checks.values()):
            raise ValueError("Replay does not match the same graph via the compiled callable")
        metrics = pair.analyze_side(folder)
        with Path(metrics["profile_csv"]).open() as stream:
            kernels = list(csv.DictReader(stream))
        streams = collections.Counter(row["Stream ID"] for row in kernels)
        super_count = sum("super" in (row["Name"] + row["Type"]).lower() for row in kernels)
        if not report["multistream"]["dsa_overlap"] or len(streams) < 2 or not super_count:
            raise ValueError("Profile does not demonstrate multistream superkernel execution")
        metrics.update(
            measurement=measurement, backend_options=options,
            samples_us=report["timing"]["samples_us"], stream_kernel_counts=dict(streams),
            superkernel_count=super_count,
            profile_device_span_us=(
                max(float(row["Start Time(us)"]) + float(row["Duration(us)"]) for row in kernels)
                - min(float(row["Start Time(us)"]) for row in kernels)),
        )
        result["cases"].append({"history": history, "batch": batch, **metrics})
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    compact_fields = ("history", "batch", "mean_us", "us_p50", "us_p95", "us_max",
                      "measurement", "samples_us", "states", "guards", "stream_kernel_counts",
                      "superkernel_count", "profile_device_span_us", "profile_json")
    summary = {**result, "cases": [{key: case[key] for key in compact_fields} for case in result["cases"]]}
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Native后端生成图的直接重放计时", "",
             "CANN9.2 / mode2 / det0，显式torch.compile backend=npugraph_ex，static和superkernel开启。",
             "每档5次预热、20次无profiler图外设备事件；profile另采，单位μs。", "",
             "| 档位 | 均值 | P50 | P95 | 最大值 | 独立profile设备跨度 | 同图状态 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for case in result["cases"]:
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {case['mean_us']:.3f} "
                     f"| {case['us_p50']:.3f} | {case['us_p95']:.3f} | {case['us_max']:.3f} "
                     f"| {case['profile_device_span_us']:.3f} | 8项零容差通过 |")
    lines += ["", "测量图由npugraph_ex生成并优化，保留两条计算stream；未捕获第二张外图。",
              "固定地址/shape、唯一后端图且无主机更新节点；八类状态与同初态的compiled callable零容差一致。",
              "报告中的eager_comparison沿用旧字段名，本轮参考实际为同一图通过编译包装调用的结果。",
              "这是CSA完整设备区间的计时校准，不是改变输入地址的生产接口或整模型验收。",
              "不再测superkernel关闭组；不同轮的调用计时与独立profile不能相减作精确开销归因。",
              "此前七档Native仍属旧编译/捕获入口，不能替换其中两行后宣称得到新版七档。", "",
              "[精简样本与检查](summary.json)、[完整配置与profile](evidence.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
