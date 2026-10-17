# SPDX-License-Identifier: Apache-2.0
"""汇总已完成的七档QK地址缓存实验；不发起测试，不混入阶段探针时间。"""

import csv
import io
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for history, batch in ((131072, 4), (131072, 8), (131072, 16), (131072, 24),
                           (8192, 16), (8192, 24), (8192, 32)):
        original = ("b16_cached_dynamic_cost_100_v1" if (history, batch) == (131072, 16)
                    else f"cached_h{history}_b{batch}_v1")
        candidate = ("b16_cached_qk_indices_v1" if (history, batch) == (131072, 16)
                     else f"qk_indices_h{history}_b{batch}_v1")
        before = json.loads((root / original / "result.json").read_text())
        after = json.loads((root / candidate / "result.json").read_text())
        for report in (before, after):
            if not (report["passed"] and report["stage"] == "complete"
                    and report["eligible_for_700us_target"] and report["progress_passed"]
                    and report["workspace_guards_passed"] and not report["stage_clock_enabled"]):
                raise ValueError("报告未通过完整检查或含探针：" + candidate)
            if (report["history"], report["batch"], report["query_tokens"]) != (history, batch, 6):
                raise ValueError("规格不一致")
            for side in ("dynamic_pto", "single_spmd"):
                values = report["l1_incore"][side]
                if len(values["samples_us"]) != 3 or abs(
                        statistics.median(values["samples_us"]) - values["median_us"]) > 1e-8:
                    raise ValueError("L1不是三个有效窗口的中位数")
        old = before["l1_incore"]["single_spmd"]["median_us"]
        current = after["l1_incore"]["single_spmd"]["median_us"]
        dynamic = after["l1_incore"]["dynamic_pto"]["median_us"]
        rows.append({"history": history, "batch": batch, "query_tokens": 6,
                     "old_single_spmd_us": old, "qk_cached_spmd_us": current,
                     "same_run_dynamic_pto_us": dynamic,
                     "change_from_old_pct": (current / old - 1) * 100,
                     "change_from_same_run_dynamic_pct": (current / dynamic - 1) * 100,
                     "new_l1_samples_us": after["l1_incore"]["single_spmd"]["samples_us"],
                     "old_l1_samples_us": before["l1_incore"]["single_spmd"]["samples_us"],
                     "event_dynamic_us": after["event_replay"]["dynamic_pto"]["median_us"],
                     "event_single_us": after["event_replay"]["single_spmd"]["median_us"],
                     "all_exact_checks_passed": True,
                     "old_result": f"{original}/result.json", "new_result": f"{candidate}/result.json"})
    report = {"scope": "正式权重、固定fixture、完整CSA单SPMD；七档分别编译",
              "timing": "完整L1 incore首尾，三个窗口中位数；无探针，不计外围调度首尾",
              "old_vs_new_is_cross_run": True, "same_run_dynamic_baseline_kept": True,
              "consecutive_decode_validated": False, "rows": rows}
    (root / "qk_indices_seven_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "qk_indices_seven_results.csv").write_text(stream.getvalue())
    lines = ["# 七档QK/PV地址缓存对比", "", "单位：μs。完整CSA的L1 incore首尾耗时，三窗中位数。",
             "旧/新单SPMD为跨轮比较；新任务各自包含原动态PTO对照，不重跑旧单SPMD。", "",
             "| 历史长度 | Batch | 原单SPMD | 新QK地址缓存 | 新同轮动态PTO | 相对原单SPMD |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['history']} | {row['batch']} | {row['old_single_spmd_us']:.2f} | "
                     f"{row['qk_cached_spmd_us']:.2f} | {row['same_run_dynamic_pto_us']:.2f} | "
                     f"{row['change_from_old_pct']:+.2f}% |")
    lines.extend(["", "七档均通过输出、Top-K、六份状态、图重放、进度代际及保护区检查。",
                  "这是固定输入恢复后的重放，不能替代连续decode轨迹验证。", "",
                  "新路径仅在当前压缩KV块内，把索引和页表载入本核UB，用Vector整数计算物理行。",
                  "Scalar从UB取地址，原逐行KV DMA、负索引分支、attention算术及FFTS保持。",
                  "Native metadata继续通过device Tensor读取，无Host回读、无额外GM展开缓冲。", "",
                  "所有任务使用各规格原有调度选择、cached等待与100tick退避。仅新增--qk-vector-indices；",
                  "没有叠加GM复用或主Q重分片。B16阶段探针另存，不进入此表或700μs门槛。", "",
                  "每档三次采样不足以证明小幅差值稳定；原始L1与Event数据保留在",
                  "[JSON](qk_indices_seven_results.json)与[CSV](qk_indices_seven_results.csv)，",
                  "两种计时范围不同，不能把L1改善直接等同于整模型收益。", ""])
    (root / "QK_SEVEN_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
