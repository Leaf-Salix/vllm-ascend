#!/usr/bin/env python3
"""只读重算Query head分组的三档完整L1，不启动任何编译/设备测试。"""

import json
from pathlib import Path

from report_prefetch_quant_round import summarize


def main():
    evidence = Path(__file__).resolve().parent / "query_head_group_evidence"
    report = summarize(evidence)
    report["decision"] = "采用通用head分组；三档中位均无回退，B24收益较一致，其余两档增益接近波动；不宣称达到30%或全动态范围已验收。"
    report["formal_changed_files"] = ["query_projection_stages.py"]
    lines = ["# Query head分组：三档完整CSA结果", "", report["measurement"], "", report["precision_scope"], "", report["decision"], "",
             "| 历史 / B | 原分task μs | d253 SPMD μs | 扩组候选 μs | 相对d253降低 | 更快窗口 | 达到30%目标 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for case in report["cases"]:
        stats = case["statistics_us"]
        lines.append(
            f"| {case['history']} / {case['batch']} | {stats['dynamic_pto']['median_us']:.2f} | "
            f"{stats['spmd_reference']['median_us']:.2f} | {stats['spmd_candidate']['median_us']:.2f} | "
            f"{case['candidate_vs_reference_reduction_pct']:.3f}% | {case['candidate_faster_windows']}/5 | "
            f"{'是' if case['target_met'] else '否'} |"
        )
    lines += ["", "正式仅迁移冻结候选的query_projection_stages.py，源按真实T整波分组，未叠加PV或其它候选。",
              "T48/144/192三个受影响回归点全部通过完整精确PTO对照、A→B→A图/状态/保护区及72核代际协议。",
              "其它四个目标点分组规则未变，本轮没有重复上卡；不把旧结果改标成新源码的完整七档复测。",
              "原始五窗、实际策略、任务与路径见同名JSON；CPU、owner模型及独立安全审阅都在本目录。", ""]
    (evidence / "DEVICE_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (evidence / "DEVICE_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
