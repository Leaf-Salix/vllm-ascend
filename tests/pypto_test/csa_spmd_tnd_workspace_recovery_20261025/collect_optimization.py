# SPDX-License-Identifier: Apache-2.0
"""复核三侧完整L1及严格正确性，将全部样本归档并更新任务状态。"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt")
    parser.add_argument("task_id")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "frozen_optimization"))
    from complete_l1_v2 import summarize

    status = subprocess.check_output(["task-submit", "--status", args.task_id], text=True).strip()
    assert status == "completed (exit=0)", status
    source = here / "runs" / args.attempt / "result.json"
    report = json.loads(source.read_text())
    assert report["passed"] and report["stage"] == "complete"
    assert len(report["checks"]) == 9 and all(report["fixture_checks"].values())
    for check in report["checks"].values():
        assert all(check["guards"].values())
        assert all(c["status"] == "PASS" and c["bit_equal"] for c in check["numerical"].values())
    assert len(report["windows"]) == 30
    samples = {side: [] for side in ("reference", "current", "candidate")}
    for window in report["windows"]:
        raw = json.loads(Path(window["raw"]).read_text())
        value = summarize(raw, init_layout=window["init_layout"])
        assert value["incore_span_us"] == window["incore_span_us"]
        assert value["task_groups"] == 1 and value["cores"] == 72
        samples[window["side"]].append(value["incore_span_us"])
    medians = {}
    for side, values in samples.items():
        assert len(values) == 10 and values == report["statistics_us"][side]["samples"]
        medians[side] = statistics.median(values)
        assert medians[side] == report["statistics_us"][side]["median"]
    summary = {
        "task_id": args.task_id,
        "attempt": args.attempt,
        "case": report["case"],
        "device": report.get("device_id"),
        "passed": True,
        "strict_check_groups": 9,
        "complete_MIX_windows": 30,
        "statistics_us": report["statistics_us"],
        "candidate_reduction_vs_current_pct": 100 * (1 - medians["candidate"] / medians["current"]),
        "candidate_reduction_vs_external_pct": 100 * (1 - medians["candidate"] / medians["reference"]),
        "paired_faster_than_current": sum(a < b for a, b in zip(samples["candidate"], samples["current"])),
        "prepare_order": report["prepare_order"],
        "adopted": False,
        "note": "同轮当前SDK对照；历史数字只作另列参照，未用作因果收益分母",
    }
    destination = here / "evidence" / (args.attempt + ".json")
    if destination.exists():
        assert destination.read_bytes() == source.read_bytes(), "拒绝改写已归档的不同结果"
    else:
        destination.write_bytes(source.read_bytes())
    name = args.attempt + "_result.json"
    (here / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    state_path = here / "current_state.json"
    state = json.loads(state_path.read_text())
    pending = [x for x in state["pending_tasks"] if x["task_id"] == args.task_id]
    if pending:
        assert len(pending) == 1
        task = pending[0]
        state["pending_tasks"].remove(task)
        task.update(status=status, result=name)
        state["completed_tasks"].append(task)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "attempt",
                    "candidate_reduction_vs_current_pct",
                    "candidate_reduction_vs_external_pct",
                    "paired_faster_than_current",
                )
            },
            ensure_ascii=False,
        )
    )
    print(json.dumps(medians, ensure_ascii=False))


if __name__ == "__main__":
    main()
