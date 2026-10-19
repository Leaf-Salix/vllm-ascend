# SPDX-License-Identifier: Apache-2.0
"""分析新正式版本的全核阶段端点；不把探针区间写入性能对照表。

phase端点是原通知前后的设备时戳。相邻端点之间包含本核计算、搬运、
Scalar、同步及探针开销；减去累计GM等待仍不等于某个硬件引擎的独占时间。
Score的READY/ACK计数由同次执行导出，只用于解释生产消费关系。全核最大
端点用于观察长尾，不能将各阶段中位数相加，也不能与旧插桩结果相减计收益。
"""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
MAIN_PHASES = (
    "PARTIAL", "QA", "QB", "KV_PROJECT", "QUERY", "IDX_SCALE",
    "SCORE_CUBE", "SCORE_VECTOR", "TOPK", "ATTN_PLAN",
    "ATTN_CUBE", "ATTN_VECTOR", "OB", "OUT",
)


def bounds(values):
    return {
        "min": round(min(values), 2),
        "median": round(median(values), 2),
        "max": round(max(values), 2),
    }


def summarize(folder):
    result = json.loads((folder / "result.json").read_text())
    assert result["passed"] and result["stage"] == "complete"
    assert result["policy"]["SPLIT_SCORE_ROWS"] is True
    assert not result["performance_measured"]
    rows = json.loads((folder / "phase_clock/phase_clock.json").read_text())
    pressure = json.loads((folder / "phase_clock/score_pressure.json").read_text())
    assert pressure["passed"] and pressure["event_counts_match_per_group"]
    cores = defaultdict(list)
    phases = defaultdict(list)
    for row in rows:
        cores[row["core"]].append(row)
    assert sorted(cores) == list(range(72))
    for core, entries in cores.items():
        ordered = sorted(entries, key=lambda r: r["notify_start_us"])
        for before, row in zip(ordered, ordered[1:]):
            interval = row["notify_start_us"] - before["notify_end_us"]
            wait = row["cumulative_wait_us"] - before["cumulative_wait_us"]
            phases[row["name"]].append({
                "core": core, "start_us": before["notify_end_us"],
                "end_us": row["notify_end_us"], "interval_us": interval,
                "gm_wait_us": wait, "other_us": interval - wait,
            })
    summary = {}
    for name, entries in phases.items():
        latest = max(entries, key=lambda r: r["end_us"])
        summary[name] = {
            "roles": len(entries), "latest_core": latest["core"],
            **{key: bounds([r[key] for r in entries])
               for key in ("start_us", "end_us", "interval_us", "gm_wait_us", "other_us")},
        }
    score_phase = {r["core"]: r for name in ("SCORE_CUBE", "SCORE_VECTOR") for r in phases[name]}
    pressure_by_core = {r["core"]: r for r in pressure["rows"]}
    groups = []
    for group in range(24):
        cube, vec0, vec1 = group, 24 + group * 2, 25 + group * 2
        groups.append({
            "group": group,
            "blocks": pressure_by_core[cube]["ack_count"],
            "cube_body_begin_estimate_us": round(
                score_phase[cube]["start_us"] + score_phase[cube]["gm_wait_us"], 2
            ),
            "vector_body_begin_estimate_us": [round(
                score_phase[v]["start_us"] + score_phase[v]["gm_wait_us"], 2
            ) for v in (vec0, vec1)],
            "cube_end_us": score_phase[cube]["end_us"],
            "vector_end_us": [score_phase[v]["end_us"] for v in (vec0, vec1)],
            "ack_us": pressure_by_core[cube]["ack_us"],
            "ready_us": [pressure_by_core[v]["ready_us"] for v in (vec0, vec1)],
        })
    return {
        "source": str(folder), "source_snapshot": "source_after_score_split",
        "source_commit": "13259f92", "passed": True,
        "performance_eligible": False, "policy": result["policy"],
        "observed_roles": 72, "phases": summary,
        "score_wait_metrics": pressure["metrics"], "score_groups": groups,
        "body_begin_estimate_note": "前一phase通知结束加累计GM等待，仅作端点估算，非直接硬件探针",
        "scope": "单次全CSA诊断；含探针/流水/Scalar，非无插桩完整L1或独占引擎时间",
    }


def main():
    tasks = json.loads((HERE / "CURRENT_SCORE_PRESSURE_TASKS.json").read_text())
    reports = {}
    for batch, task in tasks.items():
        report = summarize(HERE / task["folder"])
        report["task"] = task["task"]
        reports[batch] = report
        print("B" + batch, "Score等待", report["score_wait_metrics"])
        for phase in MAIN_PHASES:
            p = report["phases"][phase]
            print(phase, "最晚端点", p["end_us"]["max"],
                  "区间中位", p["interval_us"]["median"],
                  "GM等待中位", p["gm_wait_us"]["median"],
                  "其余中位", p["other_us"]["median"],
                  "末核", p["latest_core"])
    (HERE / "CURRENT_SCORE_PRESSURE_ANALYSIS.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
