"""只读汇总三条独立候选；以同进程正式668和原分task为准，不将异轮收益相加。"""

import argparse
import json
from pathlib import Path

from report_prefetch_quant_round import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("qb_b_resident", "qb_qr_b_resident", "oa_pair_manual", "oa_pair_small_t", "idx_had_owner"))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / f"{args.candidate}_evidence"
    report = summarize(directory)
    report["reference_commit"] = "66882634"
    report["candidate"] = args.candidate
    report["adoption"] = "本报告仅保存实验；正式采用与区间验证结论见统一优化日志。"
    lines = [f"# {args.candidate} 完整CSA同轮对照", "", report["measurement"], "",
             report["precision_scope"], "", "每行独立同进程五窗；不同候选分别运行，不比较异轮绝对时间判优。", "",
             "| 历史 / B | 原分task μs | 正式668 μs | 候选 μs | 相对668降低 | 更快窗 | 相对原分task降低 | 30%目标 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for case in report["cases"]:
        stats = case["statistics_us"]
        lines.append(f"| {case['history']} / {case['batch']} | "
                     f"{stats['dynamic_pto']['median_us']:.2f} | "
                     f"{stats['spmd_reference']['median_us']:.2f} | "
                     f"{stats['spmd_candidate']['median_us']:.2f} | "
                     f"{case['candidate_vs_reference_reduction_pct']:.3f}% | "
                     f"{case['candidate_faster_windows']}/5 | "
                     f"{case['candidate_vs_dynamic_reduction_pct']:.3f}% | "
                     f"{'达标' if case['target_met'] else '未达标'} |")
    lines += ["", report["adoption"], "", "完整样本、策略、任务与结果路径见同名JSON。", ""]
    (directory / "DEVICE_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (directory / "DEVICE_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
