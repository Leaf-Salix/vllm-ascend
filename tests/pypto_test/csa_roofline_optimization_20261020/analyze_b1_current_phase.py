# SPDX-License-Identifier: Apache-2.0
"""只分析已取得的B1全核阶段时戳，不执行NPU或将插桩区间混入性能表。"""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def main():
    here = Path(__file__).resolve().parent
    folder = here / "b1_current_phase_v1"
    result = json.loads((folder / "result.json").read_text())
    assert result["passed"] and result["phase_clock"] and not result["performance_measured"]
    rows = json.loads((folder / "phase_clock/phase_clock.json").read_text())
    origin = min(row["notify_start_us"] for row in rows if row["name"] == "BEGIN")
    cores, phases = defaultdict(list), defaultdict(list)
    for row in rows:
        cores[row["core"]].append(row)
    assert sorted(cores) == list(range(72))
    for core, entries in cores.items():
        ordered = sorted(entries, key=lambda row: row["notify_start_us"])
        for previous, row in zip(ordered, ordered[1:]):
            interval = row["notify_start_us"] - previous["notify_end_us"]
            wait = row["cumulative_wait_us"] - previous["cumulative_wait_us"]
            phases[row["name"]].append(dict(
                core=core, previous=previous["name"],
                start_us=round(previous["notify_end_us"] - origin, 2),
                end_us=round(row["notify_end_us"] - origin, 2),
                interval_us=round(interval, 2), gm_wait_us=round(wait, 2),
                other_us=round(interval - wait, 2)))
    summaries = {}
    for name, entries in phases.items():
        last = max(entries, key=lambda row: row["end_us"])
        summaries[name] = dict(roles=len(entries), latest_core=last["core"], latest_end_us=last["end_us"],
                               median_interval_us=round(median(row["interval_us"] for row in entries), 2),
                               median_gm_wait_us=round(median(row["gm_wait_us"] for row in entries), 2),
                               median_other_us=round(median(row["other_us"] for row in entries), 2))
    active_topk = [row for row in phases["TOPK"] if 24 <= row["core"] < 30]
    report = dict(passed=True, eligible_for_performance=False, observed_roles=72,
                  task="task_20261021_053711_280051422111", source="source_after_b1_half_v2",
                  wait_ticks=150, phases=summaries, all_intervals=dict(phases), active_topk=active_topk,
                  note="单次带探针端点；含冷暖状态、调度/Scalar/搬运/同步。减GM等待不等于引擎计算，不跨轮与正式L1相减。")
    (here / "B1_CURRENT_PHASE_ANALYSIS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 当前B1单SPMD阶段诊断", "", report["note"], "",
             "72个角色完整导出，完整输出/TopK/score/cache/state/保护检查通过。这里只执行一次，未做图性能采样。",
             "所有端点以本次最早BEGIN为零；480μs级结束端点不是正式无探针L1性能。", "",
             "| 阶段 | 角色数 | 最晚端点μs | 本核相邻区间中位μs | GM等待中位μs |", "|---|---:|---:|---:|---:|"]
    for name in ("META", "PARTIAL", "QA", "QB", "KV_PROJECT", "QUERY", "IDX_SCALE", "SCORE_CUBE", "SCORE_VECTOR",
                 "TOPK", "ATTN_PLAN", "ATTN_VECTOR", "OA_GROUP_BASE", "OA_GROUP_7", "OB", "OUT"):
        if name in summaries:
            s = summaries[name]
            lines.append(f"| {name} | {s['roles']} | {s['latest_end_us']:.2f} | {s['median_interval_us']:.2f} | {s['median_gm_wait_us']:.2f} |")
    lines += ["", "本次实际六个Top-K消费者，扣除已计GM等待后的区间约5～6.5μs；这个余数仍含搬运、Scalar及探针。",
              "根归并不是当前唯一优先点。输出投影串行K链、权重供给及小T尾行值得核对；不据此宣称可直接省掉某个端点差。",
              "首个低复杂度候选只让单请求T<=16的O_B取M16，保留K256/N256及反量化顺序。",
              "它减少原M32的尾行计算和INT8尾零，权重流量不变，所以不能预测耗时减半；采用由无探针完整L1决定。"]
    (here / "B1_CURRENT_PHASE_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
