"""Summarize the two Native runtime comparisons without starting the NPU."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
result = {"source_commit": "9d237d33", "cases": []}
lines = [
    "# CANN 9.0 / 9.2 Native 单卡结果",
    "",
    "同一卡、同源码、自定义算子包和 ATB；HC_pre→norm→CSA→HC_post 图重放。",
    "每侧预热 5 次、正式 20 次；下表均值/P95不含 profiler。",
    "",
    "| 档位 | 9.0 均值 μs | 9.2 均值 μs | 变化 | 9.0 P95 μs | 9.2 P95 μs |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for history in (131072, 8192):
    entry = {"history": history, "batch": 16, "versions": {}}
    for version in ("cann90", "cann92"):
        folder = ROOT / f"h{history}_b16" / version
        report = json.loads((folder / "report.json").read_text())
        timing = report["timing"]["native"]
        assert report["status"] == "MEASURED"
        assert all(x["status"] == "PASS" for x in timing["guards"].values())
        assert all(x["nonfinite"] == 0 for x in timing["eager_comparison"].values())
        assert timing["topk_selection"]["status"] != "FAIL"
        assert len(timing["samples_us"]) == 20
        kernel_files = list(folder.glob("profile/native/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        assert len(kernel_files) == 1, kernel_files
        with kernel_files[0].open() as stream:
            kernels = list(csv.DictReader(stream))
        assert any(x["Type"] == "VllmQuantLightningIndexer" for x in kernels)
        assert any(x["Type"] == "SparseAttnSharedkv" for x in kernels)
        trace = kernel_files[0].with_name("trace_view.json")
        payload = json.loads(trace.read_text())
        events = payload["traceEvents"] if isinstance(payload, dict) else payload
        assert any("LightningIndexer" in x.get("name", "") for x in events)
        entry["versions"][version] = {
            "mean_us": statistics.mean(timing["samples_us"]),
            "median_us": timing["us_p50"],
            "p95_us": timing["us_p95"],
            "max_us": timing["us_max"],
            "samples_us": timing["samples_us"],
            "guards_and_metadata": "PASS",
            "finite": "PASS",
            "topk": timing["topk_selection"],
            "provenance": report["runtime_provenance"],
            "trace": str(trace.relative_to(ROOT)),
            "single_profile_kernels": [
                {key: x[key] for key in ("Name", "Type", "Duration(us)", "aicore_time(us)", "aiv_time(us)")}
                for x in kernels
            ],
        }
    old, new = (entry["versions"][x] for x in ("cann90", "cann92"))
    entry["change_pct"] = (new["mean_us"] / old["mean_us"] - 1) * 100
    lines.append(f"| {history // 1024}K/B16 | {old['mean_us']:.3f} | {new['mean_us']:.3f} | "
                 f"{entry['change_pct']:+.3f}% | {old['p95_us']:.2f} | {new['p95_us']:.2f} |")
    result["cases"].append(entry)
lines += ["", "两档兼容性通过，均值变化不足 0.3%，没有证据表明仅切换运行库带来明显加速。",
          "不是全七档或模型验收，也没有把 release QLI/Sparse 换成 9.2 内置实现。",
          "", "## PyTorch profiling JSON", ""]
for case in result["cases"]:
    for version, values in case["versions"].items():
        lines.append(f"- [{case['history'] // 1024}K/B16 {version}]({values['trace']})")
lines += ["", "9.2 最初自动导出因安装目录属主检查失败；用本用户目录的同版本 profiler 离线重导出通过。",
          "未重跑设备采样，未更改他人安装。分算子数据来自独立一次 profile，不与20次正式均值相加或混算。", ""]
(ROOT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(ROOT / "RESULTS.md").write_text("\n".join(lines))
print("\n".join(lines[:12]))
