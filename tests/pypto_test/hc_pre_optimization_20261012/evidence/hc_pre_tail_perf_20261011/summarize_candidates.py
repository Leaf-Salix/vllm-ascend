"""汇总本目录真实 L1 候选结果；编译/精度失败与性能数据分别记录。"""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
runs = []
for report in sorted(root.glob("*/report.json")):
    data = json.loads(report.read_text())
    runs.append({"directory": report.parent.name, "status": data.get("status"),
                 "error": data.get("error"), "report": str(report)})
    if data.get("probe"):
        continue  # 带探针的泳道只用于定位，不能进入正式候选性能表。
    for case in data.get("cases", []):
        lane = case.get("l1")
        if not lane:
            continue
        if isinstance(case.get("p50_us"), dict):
            for tag, median in case["p50_us"].items():
                rows.append({"variant": report.parent.name + "/" + tag, "case": case["name"],
                             "status": data["status"], "physical_device": data.get("physical_device"),
                             "p50_us": round(median, 2),
                             "samples_us": "/".join(str(round(v, 2)) for v in case["samples_us"][tag]),
                             "trace": "|".join(lane["sources"]), "report": str(report)})
            continue
        rows.append({"variant": report.parent.name, "case": case["name"],
                     "status": data["status"], "physical_device": data.get("physical_device"),
                     "p50_us": round(lane["span_p50_us"], 2),
                     "samples_us": "/".join(str(round(s["span_us"], 2)) for s in lane["samples"]),
                     "trace": lane["source"], "report": str(report)})
(root / "candidate_status.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n")
with (root / "candidate_l1.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=["variant", "case", "status", "physical_device", "p50_us", "samples_us", "trace", "report"])
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(runs)} 份报告，{len(rows)} 组未插桩真实 L1 三次采样；探针单列")
