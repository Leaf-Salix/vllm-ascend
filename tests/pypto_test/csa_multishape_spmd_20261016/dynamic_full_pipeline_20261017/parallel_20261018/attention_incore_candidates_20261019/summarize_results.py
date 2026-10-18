#!/usr/bin/env python3
"""只读取已有三方真机结果，归档逐窗数据及门槛；不启动任何设备任务。"""

import json
from collections import Counter
from pathlib import Path


def status_counts(value):
    counts = Counter()
    if isinstance(value, dict):
        if isinstance(value.get("status"), str):
            counts[value["status"]] += 1
        for child in value.values():
            counts.update(status_counts(child))
    elif isinstance(value, list):
        for child in value:
            counts.update(status_counts(child))
    return counts


def main():
    here = Path(__file__).resolve().parent
    rows = []
    for task in json.loads((here / "DEVICE_SUBMISSIONS.json").read_text()):
        path = here / task["label"] / "result.json"
        result = json.loads(path.read_text())
        matrix = result["source_matrix"]
        stats = matrix["statistics_us"]
        candidate = stats["spmd_candidate"]["samples_us"]
        reference = stats["spmd_reference"]["samples_us"]
        rows.append({
            "label": task["label"],
            "task_id": task["stdout"].strip(),
            "history": task["history"],
            "batch": task["batch"],
            "query_tokens": task["query_tokens"],
            "source": task["source"],
            "passed": result["passed"],
            "full_csa_validated": result["full_csa_validated"],
            "graph_replay_validated": result["graph_replay_validated"],
            "check_status_counts": dict(status_counts(result)),
            "same_process": matrix["same_process"],
            "same_root_addresses": matrix["same_root_addresses"],
            "statistics_us": stats,
            "candidate_faster_windows": sum(c < r for c, r in zip(candidate, reference)),
            "candidate_vs_reference": matrix["candidate_vs_reference"],
            "candidate_vs_multitask": matrix["comparison"]["spmd_candidate"],
            "raw_result": str(path.relative_to(here)),
        })
    report = {
        "scope": "正式退避版本、完整CSA真机L1首尾；同进程同根地址三方五窗",
        "timing_unit": "us",
        "decision": "均保留为实验，不全局采用；短历史Acc收益不足以证明整个策略区间",
        "adopted": False,
        "all_cases_passed": all(row["passed"] for row in rows),
        "cases": rows,
    }
    (here / "ALL_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("候选 / 历史 / B / 多task / 当前单SPMD / 候选 / 候选较当前降低% / 更快窗口")
    for row in rows:
        s = row["statistics_us"]
        print(row["source"], row["history"], row["batch"],
              *(s[key]["median_us"] for key in ("dynamic_pto", "spmd_reference", "spmd_candidate")),
              round(row["candidate_vs_reference"]["latency_reduction_pct"], 3),
              row["candidate_faster_windows"])


if __name__ == "__main__":
    main()
