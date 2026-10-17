#!/usr/bin/env python3
"""只读重算两路独立候选；参考均为9d，不把两路或不同任务的收益相加。

每份报告仅在登记任务全部完成精度、图重放、保护区和三方五窗后生成。
本脚本不编译、不启动设备，不自动迁移候选或判断完整动态范围已验收。
"""

import argparse
import json
from pathlib import Path

from report_prefetch_quant_round import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("query_linear", "qa_small_m16_v2"))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / f"{args.candidate}_evidence"
    report = summarize(directory)
    report["reference_commit"] = "9d5327e5"
    report["candidate"] = args.candidate
    report["adoption"] = "本报告只记录证据；采用与否及边界补测见统一优化日志。"
    lines = [
        f"# {args.candidate}：独立完整CSA结果", "",
        report["measurement"], "", report["precision_scope"], "",
        "参考包含9d的Query整波分组；候选之间独立，不叠加QA或连续head分工。", "",
        "| 历史 / B | 原分task μs | 9d SPMD μs | 候选 μs | 相对9d降低 | 更快窗口 | 达到30%目标 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        stats = case["statistics_us"]
        lines.append(
            f"| {case['history']} / {case['batch']} | "
            f"{stats['dynamic_pto']['median_us']:.2f} | "
            f"{stats['spmd_reference']['median_us']:.2f} | "
            f"{stats['spmd_candidate']['median_us']:.2f} | "
            f"{case['candidate_vs_reference_reduction_pct']:.3f}% | "
            f"{case['candidate_faster_windows']}/5 | "
            f"{'是' if case['target_met'] else '否'} |"
        )
    lines += ["", report["adoption"], "",
              "原始五窗、实际策略、任务及完整结果路径保留在同名JSON。", ""]
    (directory / "DEVICE_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (directory / "DEVICE_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
