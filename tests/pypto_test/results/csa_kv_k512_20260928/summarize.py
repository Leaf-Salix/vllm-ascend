"""读取KV K512候选三档，区分核内、组跨度与完整关键链。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py"
    spec = importlib.util.spec_from_file_location("worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    result = {"operator": "30f2b228 + fixed-K KV K512; atomic0",
              "tasks": ["task_20260928_083347_173445228283", "task_20260928_090325_18866938672"],
              "scope": "三档同配置20次无profiler图计时、各2个独立DFX窗口；layer4真实权重、合成历史。",
              "limits": "核内时间包含DMA和等待，总核时间不是完整跨度；本体/Native控制与核内分别报告。"
                        "逐元素比较8类状态，不含idx_topk_scores；不是EP16验收。", "cases": []}
    for history, batch in ((131072, 16), (8192, 16), (8192, 40)):
        folder = ROOT / f"h{history}_b{batch}"
        comparison = json.loads((folder / "comparison.json").read_text())
        if comparison["status"] != "PASS":
            raise ValueError(comparison["errors"])
        case = {"history": history, "batch": batch, "state_check": "PASS", "measurements": {}}
        for side in ("baseline", "candidate"):
            timing_path = folder / side / "report.json"
            report = json.loads(timing_path.read_text())
            swimlane_path = folder / "swimlane" / side / "report.json"
            if side == "baseline" and not swimlane_path.exists():
                swimlane_path = (ROOT.parent / "csa_qa_adaptive_20260928" /
                                 folder.name / "swimlane/candidate/report.json")
            swimlane = json.loads(swimlane_path.read_text())
            windows = [worker.summarize(Path(w["merged_swimlane"])) for w in swimlane["swimlane_windows"]]
            if len(windows) != 2 or report["timing"]["iters"] != 20:
                raise ValueError(f"{history}/{side}: 样本不完整")
            kv = [w["tasks"]["kv_proj_matmul"] for w in windows]
            case["measurements"][side] = {
                "timing_source": str(timing_path), "swimlane_source": str(swimlane_path),
                "timing": {backend: {
                    "samples_us": report["timing"][backend]["samples_us"],
                    "mean_us": statistics.mean(report["timing"][backend]["samples_us"]),
                    "p95_us": report["timing"][backend]["us_p95"],
                    "max_us": report["timing"][backend]["us_max"]} for backend in ("native", "pto")},
                "worker_windows": windows,
                "kv_windows": [{**v, "summed_kernel_us": v["kernel_mean_us"] * v["blocks"]} for v in kv],
            }
        before, after = (case["measurements"][s] for s in ("baseline", "candidate"))
        case["change_pct"] = {
            backend: (after["timing"][backend]["mean_us"] / before["timing"][backend]["mean_us"] - 1) * 100
            for backend in ("native", "pto")
        }
        for key in ("kernel_mean_us", "summed_kernel_us", "worker_envelope_us"):
            case["change_pct"][key] = (statistics.mean(w[key] for w in after["kv_windows"]) /
                                      statistics.mean(w[key] for w in before["kv_windows"]) - 1) * 100
        result["cases"].append(case)
        print(history, batch, case["change_pct"])
    (ROOT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
