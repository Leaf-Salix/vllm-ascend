#!/usr/bin/env python3
"""汇总同一套V10算子代码的实测结果；保留各档实际采集来源。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "csa_native_cube_matrix_20260927"
CASES = (
    (131072, 4, "v10_native_balance", "0ed4f926"),
    (131072, 8, "v10_unified_followup", "79aaed98"),
    (131072, 16, "v10_unified_followup", "79aaed98"),
    (8192, 16, "v10_short_followup", "0ed4f926"),
    (8192, 24, "v10_unified_followup", "79aaed98"),
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
            "implementation": "v10",
            "operator_revision": "0ed4f926",
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
        "# 统一V10实现：七档单卡实测（2026-09-27）",
        "",
        "七档均实测同一套V10性能版算子源码（0ed4f926），长短上下文与batch策略由算子内部选择。",
        "V10累计保留两query共享key/M128 QK、8K片上FP16/Cube规约和小batch工作量均衡。",
        "128K/B8、B16及8K/B24补测使用checkout 79aaed98，其生产算子源码与0ed4f926无差异；",
        "其余四档复用已经实测的V10记录。label仅标识采集目录，revision记录采集checkout。",
        "数据来自不同采集批次，Native列为各行同轮测量；不再用V8/V9测量代替V10实测。",
        "旧七档V4/V7基线独立保留，编译失败的V11未合入正式入口。",
        "正式layer 4权重、合成输入和历史，单卡S6/TP1/mode2/atomic1/确定性0/EPLB关闭。",
        "复用第二个CSA层metadata，5次预热、20次无profiler采样。单位μs。",
        "本体含HC_pre→norm→CSA→HC_post，完整PTO另含拆分和slot写回；各阶段独立测量。",
        "",
        "| H / B | 实现 | 同轮Native | v7本体 | V10本体 | 对v7 | 对Native | 本体p50 / p95 | 完整PTO |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        timing = row["timing"]
        body = timing["body"]
        mean, native, old = body["mean_us"], timing["native"]["mean_us"], row["v7"]["body"]["mean_us"]
        lines.append(
            f"| {row['history'] // 1024}K / {row['batch']} | V10 | {native:.2f} | {old:.2f} | "
            f"{mean:.2f} | {100 * (mean / old - 1):+.2f}% | {100 * (mean / native - 1):+.2f}% | "
            f"{body['p50_us']:.2f} / {body['p95_us']:.2f} | {timing['pto_full']['mean_us']:.2f} |"
        )
    lines += [
        "",
        "统一策略由`decode_indexer.py`根据输入决定，调用者不选择V8/V9/V10：",
        "",
        "| 输入条件 | 算子内策略 |",
        "| --- | --- |",
        "| 本batch最大压缩历史≥2048行 | 两query共享key、M128 QK、片上FP16 Score与Cube head规约；"
        "本矩阵8K/128K均走此路 |",
        "| 本batch最大压缩历史<2048行 | 保留较短历史的Vector规约路径 |",
        "| Cube路径query总数<48（S6时B<8） | leaf优先分派，均衡完整leaf工作量；本矩阵128K/B4命中 |",
        "| Cube路径query总数≥48 | query组优先分派；其余六档命中 |",
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
