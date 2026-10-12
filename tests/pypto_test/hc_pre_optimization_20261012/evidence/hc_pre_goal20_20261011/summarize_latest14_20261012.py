"""汇总当前正式 HC_pre 的 14 档真实 L1 结果，不读取旧版取消的 ABBA。"""
import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


root = Path(__file__).resolve().parent
manifest = json.loads((root / "latest14_20261012_manifest.json").read_text())
snapshot = (root / manifest["snapshot"]).read_bytes()
rows = []
for item in manifest["rows"]:
    report_path = root / item["label"] / "report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "PASS", (report_path, report["status"])
    assert Path(report["kernel_file"]).read_bytes() == snapshot, report_path
    assert report["swimlane_level"] == 1 and report["warmup"] == 5
    assert report["pto_warmup_after_native_profile"] == 5
    assert report["device_execution"] is True
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["status"] == "PASS" and case["output_guards"]
    assert (case["batch_size"], case["seqlen"]) == (item["batch_size"], item["seqlen"])
    for key in ("pto_native", "native_graph_eager", "pto_graph_eager",
                "native_updated_input", "pto_updated_input"):
        assert all(value["passed"] for value in case[key].values()), (report_path, key)
    timing = case["incore_timing"]
    pto, native = timing["pto"], timing["native"]
    assert pto["steps"] == native["steps"] == 3
    assert len(pto["samples"]) == 3
    for sample in pto["samples"]:
        assert sample["blocks"] == 72
        assert sorted(task["blocks"] for task in sample["tasks"].values()) == [24, 48]
    samples = [sample["span_us"] for sample in pto["samples"]]
    rows.append({
        "batch_size": case["batch_size"], "seqlen": case["seqlen"], "tokens": case["tokens"],
        "variant": case["variant"]["name"], "status": "PASS",
        "native_p50_us": native["p50_us"], "pto_l1_p50_us": pto["span_p50_us"],
        "pto_l1_min_us": min(samples), "pto_l1_max_us": max(samples),
        "pto_l1_samples_us": samples, "native_samples_us": native["samples_us"],
        "native_over_pto": native["p50_us"] / pto["span_p50_us"],
        "physical_device": report.get("physical_device"), "reused_same_day": item["reused"],
        "report_time": datetime.fromtimestamp(report_path.stat().st_mtime,
                         timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "report": str(report_path.relative_to(root)),
        "pto_trace": pto["source"], "native_trace": native["source"],
    })
assert {(r["batch_size"], r["seqlen"]) for r in rows} == {
    (b, s) for b in [2, 4, 8, 16, 20, 24, 28] for s in [5, 6]
}
output = {
    "status": "PASS", "scope": manifest["scope"], "snapshot": manifest["snapshot"],
    "metric": manifest["metric"], "jobs": manifest["jobs"], "rows": rows,
    "precision": "沿用既有门禁；初始/同址换输入、graph/eager、保护行均通过。",
    "note": "3 档复用本轮刚完成的同源码同设置对照，11 档补测；所有数据来自 2026-10-12。新候选未混入。",
}
(root / "latest14_20261012_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
with (root / "latest14_20261012_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ["# 当前正式 HC_pre：14 档性能（2026-10-12）", "", output["scope"], "",
         output["note"], "", "计时：" + output["metric"] + "。Native 列为 HcPre 设备事件耗时。",
         "每档均通过原有精度门禁、graph/eager、同址换输入与保护行检查。", "",
         "| B | S | T | Native μs | PTO L1 μs | PTO 三次范围 μs | Native/PTO |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f'| {r["batch_size"]} | {r["seqlen"]} | {r["tokens"]} | '
                 f'{r["native_p50_us"]:.2f} | {r["pto_l1_p50_us"]:.2f} | '
                 f'{r["pto_l1_min_us"]:.2f}–{r["pto_l1_max_us"]:.2f} | {r["native_over_pto"]:.2f}× |')
lines += ["", "表中为三次中位数，不包含外层编排首尾，但保留 incore 开始后的全部等待。",
          "不代表整模型性能；不把不同卡、不同形状的相对变化解释为优化收益。", "",
          "源码快照：[latest14_formal_20261012.py](latest14_formal_20261012.py)。",
          "完整采样、卡号、时间、报告和泳道路径见 [JSON](latest14_20261012_summary.json)；"
          "表格另存为 [CSV](latest14_20261012_summary.csv)。", ""]
(root / "latest14_20261012_summary.md").write_text("\n".join(lines))
manifest["status"] = "PASS"
manifest["summary"] = "latest14_20261012_summary.json"
(root / "latest14_20261012_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print("\n".join(lines))
