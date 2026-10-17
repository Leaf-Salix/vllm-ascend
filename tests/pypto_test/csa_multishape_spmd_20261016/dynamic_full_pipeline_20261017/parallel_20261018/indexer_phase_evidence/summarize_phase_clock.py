# SPDX-License-Identifier: Apache-2.0
"""只读阶段原始时钟并输出诊断统计；不把插桩端点写成正式性能收益。"""

import argparse
import json
import statistics
from pathlib import Path

FOCUS_PHASES = (30, 32, 34, 35, 36, 37, 38, 39, 43)


def describe(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def summarize(directory, destination):
    report = json.loads((directory / "result.json").read_text())
    assert report["passed"] and report["full_csa_validated"]
    summary = dict(
        passed=True,
        eligible_for_performance=False,
        source=str(directory),
        history=report["history"],
        batch=report["batch"],
        rounds=[],
        paired_deltas=[],
        units="us, 除明确标为 calls 的计数",
        limitations=[
            "阶段窗口含插桩开销，不能作为完整 L1 收益或代替未插桩性能结果",
            "GM wait 仅 TWAIT；ffts_issue 为 wait_flag_dev 原位计时且未增加 fence",
            "other 包含 FFTS、搬运、计算和插桩，不作纯计算解释",
            "全核最大端点差不是各核 interval 的求和，也不代表每核都在执行该阶段",
            "AIC phase34 前一个发布点是 phase30；AIC phase38 前一个发布点是 phase34",
        ],
    )
    tokens = report["batch"] * report["query_tokens"]
    # 对应 wait_row_owners(...,0,tokens,8,48,24,1)，只保留 plan 真正 owner。
    plan_owners = {24 + group % 48 for group in range((tokens + 7) // 8)}
    for item in report["clock_rounds"]:
        rows = json.loads(Path(item["clock"]["raw"]).read_text())
        groups = {}
        for row in rows:
            groups.setdefault(row["phase"], []).append(row)
        phase_stats = {}
        for phase in FOCUS_PHASES:
            if phase not in groups:
                continue
            points = groups[phase]
            stat = dict(
                name=points[0]["name"],
                cores=len(points),
                notify_start_us=describe([r["notify_start_us"] for r in points]),
                notify_end_us=describe([r["notify_end_us"] for r in points]),
                latest_core=max(points, key=lambda r: r["notify_end_us"]),
            )
            for key in ("interval_us", "wait_us", "other_us", "ffts_issue_us", "ffts_issue_calls"):
                values = [r[key] for r in points if key in r]
                if values:
                    stat[key] = describe(values)
            phase_stats[str(phase)] = stat
        ends = {phase: max(r["notify_end_us"] for r in points) for phase, points in groups.items()}
        plan_end = max(r["notify_end_us"] for r in groups[37] if r["core"] in plan_owners)
        windows = dict(
            input_to_scale_end=ends[32],
            scale_to_score_cube_end=ends[34] - ends[32],
            scale_to_score_vector_end=ends[35] - ends[32],
            score_vector_tail_after_cube=ends[35] - ends[34],
            score_vector_to_topk_end=ends[36] - ends[35],
            topk_to_plan_owner_end=plan_end - ends[36],
            plan_owner_to_attention_cube_end=ends[38] - plan_end,
            plan_owner_to_attention_vector_end=ends[39] - plan_end,
            attention_vector_to_output_end=ends[43] - ends[39],
            scale_to_plan_owner_end=plan_end - ends[32],
            scale_to_attention_vector_end=ends[39] - ends[32],
            instrumented_last_output_endpoint=ends[43],
        )
        summary["rounds"].append(
            dict(
                round=item["round"],
                side=item["side"],
                label=item["label"],
                phases=phase_stats,
                windows_us=windows,
                plan_owner_cores=sorted(plan_owners),
                plan_owner_end_us=plan_end,
            )
        )
    for round_index in range(report["rounds"]):
        pair = {r["side"]: r for r in summary["rounds"] if r["round"] == round_index}
        if "spmd_candidate" not in pair:
            continue
        old, new = pair["spmd_reference"], pair["spmd_candidate"]
        summary["paired_deltas"].append(
            dict(
                round=round_index,
                label=old["label"],
                meaning="candidate - reference; positive 表示候选更长/更晚",
                windows_us={key: new["windows_us"][key] - value for key, value in old["windows_us"].items()},
                phase_medians={
                    phase: {
                        key: new["phases"][phase][key]["median"] - old["phases"][phase][key]["median"]
                        for key in ("interval_us", "wait_us", "other_us", "ffts_issue_us", "ffts_issue_calls")
                        if key in old["phases"][phase] and key in new["phases"][phase]
                    }
                    for phase in old["phases"]
                },
            )
        )
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(destination)
    for row in summary["rounds"]:
        print(json.dumps({key: row[key] for key in ("round", "side", "label", "windows_us")}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.output or args.directory / "phase_summary.json")


if __name__ == "__main__":
    main()
