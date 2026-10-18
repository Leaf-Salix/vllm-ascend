#!/usr/bin/env python3
"""汇总两版动态SPMD的同进程阶段端点；插桩数据仅定位，不用于性能验收。

每轮以本版所有核最早BEGIN为相对零点。端点是同一phase所有发布者中
最晚的notify完成，不等于数学执行时长；重排后空发布者和真实producer的
数量不同，不能直接比较所有核other_us的中位数来断言算术快慢。
"""

import json
import statistics
from pathlib import Path


PHASES = (
    "IDX_QUERY_PROJECT", "IDX_NORM", "IDX_QUERY_ROPE", "WEIGHTS_PROJECT",
    "IDX_QUERY_HAD", "IDX_HADAMARD", "KV_PROJECT", "ORIGINAL_CACHE",
    "IDX_QUERY_QUANT", "COEFF", "IDX_SCALE", "QB", "QUERY", "MAIN_PROJECT",
    "MAIN_OUTPUT", "SCORE_CUBE", "SCORE_VECTOR", "TOPK", "ATTN_CUBE", "OUT",
)


def main():
    here = Path(__file__).resolve().parent
    path = here / "h131072_b16_branch4_clock_v1/result.json"
    result = json.loads(path.read_text())
    if not result["passed"] or result["stage"] != "complete":
        raise ValueError("阶段任务尚未完成，不能汇总为通过")
    collected = {side: {phase: [] for phase in PHASES}
                 for side in ("spmd_reference", "spmd_candidate")}
    rounds = []
    for run in result["clock_rounds"]:
        rows = json.loads(Path(run["clock"]["raw"]).read_text())
        begin = min(row["notify_start_us"] for row in rows if row["name"] == "BEGIN")
        ends = {}
        for phase in PHASES:
            values = [row["notify_end_us"] for row in rows if row["name"] == phase]
            if not values:
                raise ValueError(f"缺少阶段：{phase}")
            ends[phase] = max(values) - begin
            collected[run["side"]][phase].append(ends[phase])
        rounds.append({"round": run["round"], "side": run["side"], "input": run["label"],
                       "raw": run["clock"]["raw"], "latest_notify_end_from_begin_us": ends})
    summary = []
    for phase in PHASES:
        old = statistics.median(collected["spmd_reference"][phase])
        new = statistics.median(collected["spmd_candidate"][phase])
        summary.append({"phase": phase, "reference_end_us": old, "candidate_end_us": new,
                        "candidate_minus_reference_us": new - old})
    report = {
        "task": "task_20261018_115704_29584807981", "source": str(path),
        "passed": True, "rounds_per_side": 3, "inputs": ["A", "B", "A"],
        "same_process": True, "same_native_root_addresses": True,
        "performance_acceptance": False,
        "metric": "各版所有核最早BEGIN到该phase最后一次notify结束；三轮中位数",
        "limitation": "插桩改变时序；端点是就绪时间，不是数学耗时；other包含同步和搬运",
        "phase_ends": summary, "rounds": rounds,
        "finding": "Indexer norm和主压缩更早完成，但4核KV分支明显延后；AIV的KV前置顺序阻挡后续Indexer量化和Score",
        "adopt_candidate": False,
        "next_constraint": "调整分工必须同时覆盖AIC与AIV关键消费者顺序，不能只增加辅助核数量盲测",
    }
    destination = here / "PHASE_CLOCK_RESULT_V1.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in summary:
        print(f"{row['phase']:20s} {row['reference_end_us']:8.2f} {row['candidate_end_us']:8.2f} "
              f"{row['candidate_minus_reference_us']:8.2f}")


if __name__ == "__main__":
    main()
