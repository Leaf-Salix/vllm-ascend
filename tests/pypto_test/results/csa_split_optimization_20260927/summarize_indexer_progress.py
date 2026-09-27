#!/usr/bin/env python3
"""汇总已保留Indexer改动；复用未受影响档位，不重新占卡。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "csa_native_cube_matrix_20260927"
CASES = (
    (131072, 4, "v10_native_balance", "0ed4f926"),
    (131072, 8, "v8_native_pair", "05e0b518"),
    (131072, 16, "v8_native_pair", "05e0b518"),
    (8192, 16, "v10_short_followup", "0ed4f926"),
    (8192, 24, "v9_native_short", "c553120c"),
    (8192, 32, "v10_short_followup", "0ed4f926"),
    (8192, 40, "v10_short_followup", "0ed4f926"),
)


def main():
    spec = importlib.util.spec_from_file_location("matrix_summary", BASELINE / "summarize.py")
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    rows = []
    for history, batch, label, revision in CASES:
        case = ROOT / label / f"h{history}_b{batch}"
        old = (
            ROOT / "v7_native_overlap" if (history, batch) == (131072, 16) else BASELINE / "v7"
        ) / f"h{history}_b{batch}/timing/report.json"
        row = {
            "history": history,
            "batch": batch,
            "label": label,
            "revision": revision,
            "timing": summary.summarize_timing(case / "timing/report.json"),
            "v7": summary.summarize_timing(old),
            "swimlane": [
                summary.summarize_swimlane(path)
                for path in sorted((case / "swimlane/dfx").glob("**/merged_swimlane.json"))
            ],
        }
        rows.append(row)
    (ROOT / "indexer_progress_v10.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# 已保留Indexer优化：v8～v10阶段结果（2026-09-27）",
        "",
        "旧七档v4/v7矩阵保持独立；本表汇总后续已保留实现，不含编译失败的v11。",
        "复用未受后续改动影响的v8长上下文B8/B16、v9短上下文B24；其余使用v10。",
        "这不是统一重跑的严格A/B，Native列为各行本次测量的同轮Native；源码和原始报告见JSON。",
        "正式layer 4权重、合成输入和历史，单卡S6/TP1/mode2/atomic1/确定性0/EPLB关闭。",
        "复用第二个CSA层metadata，5次预热、20次无profiler采样。单位μs。",
        "本体含HC_pre→norm→CSA→HC_post，完整PTO另含拆分和slot写回；各阶段独立测量。",
        "",
        "| H / B | 候选 | 同轮Native | v7本体 | 本次本体 | 对v7 | 对Native | 本体p50 / p95 | 完整PTO |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        timing = row["timing"]
        body = timing["body"]
        mean, native, old = body["mean_us"], timing["native"]["mean_us"], row["v7"]["body"]["mean_us"]
        lines.append(
            f"| {row['history'] // 1024}K / {row['batch']} | {row['label']} | {native:.2f} | {old:.2f} | "
            f"{mean:.2f} | {100 * (mean / old - 1):+.2f}% | {100 * (mean / native - 1):+.2f}% | "
            f"{body['p50_us']:.2f} / {body['p95_us']:.2f} | {timing['pto_full']['mean_us']:.2f} |"
        )
    lines += [
        "",
        "PTO泳道独立采集；Score→publish含调度和交叠，不等于独占算术耗时。",
        "",
        "| H / B | Score核内均值范围 | Score→publish范围 | 输出RMSE | max_abs | Top-K集合替换 | 保护区失败数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        timing, windows = row["timing"], row["swimlane"]
        incore = [window["incore_mean_us"] for window in windows]
        span = [window["score_to_publish_us"] for window in windows]
        incore_text = f"{min(incore):.2f}～{max(incore):.2f}" if incore else "未补采"
        span_text = f"{min(span):.2f}～{max(span):.2f}" if span else "未补采"
        lines.append(
            f"| {row['history'] // 1024}K / {row['batch']} | {incore_text} | "
            f"{span_text} | {timing['output']['rmse']:.7f} | "
            f"{timing['output']['max_abs']:.7f} | {timing['topk']['replaced_indices']} | "
            f"{len(timing['guard_failures'])} |"
        )
    lines += [
        "",
        "浮点零容差仍为FAIL，仅用来保留误差证据。结构/保护区通过不能替代数值验收。",
        "新FP16/Top-K策略尚未做真实权重16卡逐token/DSpark验收；<750μs目标未完成。",
        "128K/B16仍有长尾，不能只用正常窗口或均值宣布Indexer已稳定赶上Native。",
        "[逐项数据和原始路径](indexer_progress_v10.json)；",
        "[之前七档v4/v7](../csa_native_cube_matrix_20260927/README.md)。",
    ]
    (ROOT / "INDEXER_PROGRESS_V10.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
