"""汇总QR分组核内与本体；只读并复用同基底的已有DFX。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    result = {
        "operator": "88d0744f + adaptive fixed-K QR candidate.patch",
        "task": "task_20260928_070150_117104915830",
        "scope": "layer4真实权重、合成历史、可变物理行scale；B16长短、B40计时，B4仅状态；"
                 "长/B40基线DFX复用刚结束的同版本对照。",
        "limits": "8类状态不含idx_topk_scores；精确比较修改前后，不是Native零误差。"
                  "DFX与计时独立，核内含DMA和等待，累计核时间不能相加当完整跨度。",
        "cases": [],
    }
    for history, batch in ((8192, 16), (8192, 40), (131072, 16), (8192, 4)):
        folder = ROOT / f"h{history}_b{batch}"
        comparison = json.loads((folder / "comparison.json").read_text())
        if comparison["status"] != "PASS":
            raise ValueError(f"B{batch}: {comparison['errors']}")
        case = {"history": history, "batch": batch, "state_check": "PASS", "measurements": {}}
        for side in ("baseline", "candidate"):
            path = folder / side / "report.json"
            report = json.loads(path.read_text())
            measurement = {"source": str(path)}
            if batch != 4:
                if report["timing"]["iters"] != 20:
                    raise ValueError(f"B{batch}/{side}: 计时数量不符")
                measurement["timing"] = {
                    backend: {
                        "samples_us": report["timing"][backend]["samples_us"],
                        "mean_us": statistics.mean(report["timing"][backend]["samples_us"]),
                        "p95_us": report["timing"][backend]["us_p95"],
                        "max_us": report["timing"][backend]["us_max"],
                    } for backend in ("native", "pto")}
            if batch != 4:
                path = folder / "swimlane" / side / "report.json"
                if side == "baseline" and (history, batch) != (8192, 16):
                    path = ROOT.parent / "csa_kv_sync_20260928" / folder.name / "swimlane/baseline/report.json"
                swimlane = json.loads(path.read_text())
                windows = [worker.summarize(Path(w["merged_swimlane"]))
                           for w in swimlane["swimlane_windows"]]
                if len(windows) != 2:
                    raise ValueError(f"B{batch}/{side}: DFX窗口数量不符")
                measurement["swimlane_source"] = str(path)
                measurement["worker_windows"] = windows
                measurement["qr_windows"] = [
                    {**w["tasks"]["qr_proj_matmul"], "path": w["path"],
                     "summed_kernel_us": w["tasks"]["qr_proj_matmul"]["kernel_mean_us"] *
                     w["tasks"]["qr_proj_matmul"]["blocks"],
                     "worker_span_us": w["worker_span_us"]} for w in windows]
            case["measurements"][side] = measurement
        if batch != 4:
            before, after = (case["measurements"][side] for side in ("baseline", "candidate"))
            case["change_pct"] = {
                backend: (after["timing"][backend]["mean_us"] /
                          before["timing"][backend]["mean_us"] - 1) * 100
                for backend in ("native", "pto")}
            for key in ("kernel_mean_us", "summed_kernel_us", "worker_envelope_us"):
                case["change_pct"][key] = (
                    statistics.mean(w[key] for w in after["qr_windows"]) /
                    statistics.mean(w[key] for w in before["qr_windows"]) - 1) * 100
        result["cases"].append(case)
        print(history, batch, case.get("change_pct", "state only"))
    (ROOT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
