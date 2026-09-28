"""比较HC融合的累计核内工作、关键链与独立无profiler计时。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    result = {
        "operator": "e58ddc94 vs HC widen+RMS fusion, atomic0/mode2/det1/layer4",
        "task": "task_20260928_103131_11716825293",
        "scope": "各档20次同轮图计时、各侧2个独立DFX窗口；核内累计工作不是墙钟耗时。",
        "limits": "输入是真实层权重与合成历史；8类状态不含idx_topk_scores，不代表整网token验收。",
        "cases": [],
    }
    lines = ["# HC输入加宽与RMS融合实测", "", result["scope"], "", result["limits"], "",
             "| 上下文/B | 项目 | 基线 μs | 融合 μs | 变化 |", "| --- | --- | ---: | ---: | ---: |"]
    for history in (131072, 8192):
        folder = ROOT / f"h{history}_b16"
        comparison = json.loads((folder / "comparison.json").read_text())
        case = {"history": history, "batch": 16, "state_check": comparison, "measurements": {}}
        for side in ("baseline", "candidate"):
            timing_path = folder / side / "report.json"
            report = json.loads(timing_path.read_text())
            swimlane_path = folder / "swimlane" / side / "report.json"
            reused = not swimlane_path.exists()
            if reused:
                if side != "baseline":
                    raise FileNotFoundError(swimlane_path)
                swimlane_path = ROOT.parent / f"csa_qr_k512_20260928/h{history}_b16/swimlane/baseline/report.json"
            swimlane = json.loads(swimlane_path.read_text())
            windows = [worker.summarize(Path(w["merged_swimlane"])) for w in swimlane["swimlane_windows"]]
            if len(windows) != 2 or report["timing"]["iters"] != 20:
                raise ValueError(f"{history}/{side}: 样本不完整")
            hc = []
            names = ("hc_widen", "hc_pre_rms") if side == "baseline" else ("hc_widen_rms",)
            for window in windows:
                tasks = window["tasks"]
                hc.append({
                    "input_rms_summed_kernel_us": sum(tasks[n]["kernel_mean_us"] * tasks[n]["blocks"] for n in names),
                    "first_worker_to_norm_end_us": window["phases_us"]["first_worker_to_norm_end"],
                    "linear_first_start_us": tasks["hc_pre_linear"]["first_start_us"],
                    "linear_last_end_us": tasks["hc_pre_linear"]["last_end_us"],
                    "linear_kernel_mean_us": tasks["hc_pre_linear"]["kernel_mean_us"],
                })
            case["measurements"][side] = {
                "timing_source": str(timing_path), "swimlane_source": str(swimlane_path),
                "swimlane_reused": reused,
                "timing": {backend: {
                    "samples_us": report["timing"][backend]["samples_us"],
                    "mean_us": statistics.mean(report["timing"][backend]["samples_us"]),
                    "p50_us": report["timing"][backend]["us_p50"],
                    "p95_us": report["timing"][backend]["us_p95"],
                    "max_us": report["timing"][backend]["us_max"],
                } for backend in ("native", "pto")},
                "worker_windows": windows, "hc_windows": hc,
            }
        before, after = (case["measurements"][s] for s in ("baseline", "candidate"))
        case["change_pct"] = {}
        values = [("本体 " + k, before["timing"]["pto"][k], after["timing"]["pto"][k])
                  for k in ("mean_us", "p50_us", "p95_us", "max_us")]
        values += [(key, statistics.mean(w[key] for w in before["hc_windows"]),
                    statistics.mean(w[key] for w in after["hc_windows"])) for key in hc[0]]
        values += [("Native控制均值", before["timing"]["native"]["mean_us"], after["timing"]["native"]["mean_us"])]
        for name, a, b in values:
            change = (b / a - 1) * 100
            case["change_pct"][name] = change
            lines.append(f"| {history}/16 | {name} | {a:.2f} | {b:.2f} | {change:+.2f}% |")
        result["cases"].append(case)
        print(json.dumps({"history": history, "state": comparison["status"], "change_pct": case["change_pct"]}, ensure_ascii=False))
    (ROOT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
