# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Analyze fixed-chain diagnostic timestamps without treating them as benchmark samples."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

TICKS_PER_US = 50
CAN_HANDOFF = 1
CAN_EARLY_START = 4


def distribution(values):
    return {"count": len(values), "min": min(values), "mean": mean(values), "max": max(values)}


def analyze_run(run):
    entries = run["entries"]
    by_group = defaultdict(list)
    by_core = defaultdict(list)
    identities = set()
    core_mapping = {}
    for e in entries:
        identity = e["stage"], e["block"]
        assert identity not in identities, f"duplicate stage/block: {identity}"
        identities.add(identity)
        assert e["prepare_start"] <= e["prepare_end"] <= e["publish_before"] <= e["publish_after"]
        assert e["start"] <= e["returned"] <= e["end"]
        assert core_mapping.setdefault(e["logical_core"], e["core"]) == e["core"]
        by_group[e["stage"]].append(e)
        by_core[e["core"]].append(e)
    assert len(set(core_mapping.values())) == len(core_mapping)
    assert entries and sorted(by_group) == list(range(len(by_group)))

    segments = {
        "cpu_prepare_us": ("prepare_start", "prepare_end"),
        "prepared_to_publish_us": ("prepare_end", "publish_before"),
        "cpu_publish_bracket_us": ("publish_before", "publish_after"),
        "publish_before_to_function_trace_us": ("publish_before", "start"),
        "publish_after_to_function_trace_us": ("publish_after", "start"),
    }
    durations = {
        name: distribution([(e[end] - e[begin]) / TICKS_PER_US for e in entries])
        for name, (begin, end) in segments.items()
    }
    local_pairs = []
    for core, chain in by_core.items():
        ordered = sorted(chain, key=lambda e: e["dispatch_id"])
        assert len({e["dispatch_id"] for e in chain}) == len(chain)
        for previous, current in zip(ordered, ordered[1:]):
            if not (current["pending"] and current["prefix"] & CAN_HANDOFF):
                continue
            overlap = current["start"] < previous["end"]
            if overlap:
                category = "start_before_local_tail_record"
            elif current["publish_before"] >= previous["end"]:
                category = "publish_started_after_local_tail_record"
            elif current["publish_after"] <= previous["returned"]:
                category = "published_before_return_but_no_observed_overlap"
            else:
                category = "publication_or_entry_in_tail_interval"
            local_pairs.append(
                dict(
                    core=core,
                    previous=[previous["stage"], previous["block"]],
                    current=[current["stage"], current["block"]],
                    category=category,
                    publish_after_minus_previous_return_us=(current["publish_after"] - previous["returned"])
                    / TICKS_PER_US,
                    start_minus_previous_end_us=(current["start"] - previous["end"]) / TICKS_PER_US,
                    previous_recorded_tail_us=(previous["end"] - previous["returned"]) / TICKS_PER_US,
                )
            )
    categories = defaultdict(int)
    for pair in local_pairs:
        categories[pair["category"]] += 1

    groups = []
    for stage, group in sorted(by_group.items()):
        if stage == 0:
            continue
        previous_end = max(e["end"] for e in by_group[stage - 1])
        terminal = max(group, key=lambda e: e["end"])
        groups.append(
            dict(
                stage=stage,
                tasks=len(group),
                published_before_previous_group_tail=sum(e["publish_after"] < previous_end for e in group),
                started_before_previous_group_tail=sum(e["start"] < previous_end for e in group),
                gated_early_start_capable=sum(
                    e["gated"] and bool(e["prefix"] & CAN_EARLY_START) for e in group
                ),
                terminal_block=terminal["block"],
                terminal_publish_before_minus_previous_end_us=(terminal["publish_before"] - previous_end)
                / TICKS_PER_US,
                terminal_start_minus_previous_end_us=(terminal["start"] - previous_end) / TICKS_PER_US,
            )
        )
    return dict(
        work=run["work"],
        entries=len(entries),
        physical_cores=len(by_core),
        pending=sum(e["pending"] for e in entries),
        gated=sum(e["gated"] for e in entries),
        timestamp_order_violations=sum(e["start"] < e["publish_before"] for e in entries),
        diagnostic_segments=durations,
        eligible_local_pair_categories=dict(categories),
        local_pairs=local_pairs,
        groups=groups,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    assert source["diagnostic_only"] is True and source["output_and_guards_exact"] is True
    assert len(source["runs"]) == 2 and {r["work"] for r in source["runs"]} == {5, 167}
    for run in source["runs"]:
        assert {(e["stage"], e["block"]) for e in run["entries"]} == {
            (stage, block) for stage in range(17) for block in range(48)
        }
    report = dict(
        diagnostic_only=True,
        limitations=[
            "Instrumentation changes scheduling; these are not uninstrumented latency samples.",
            "Tail end includes the notification and runtime PIPE_ALL; it is not original Vector completion.",
            "Publish brackets are CPU observations, not a measured device-register visibility time.",
            "Function trace excludes some wrapper work; delay includes core pickup and argument preparation.",
            "A function may enter before publish_after; negative after-to-entry values are retained.",
            "No overlap after early publication requires inspection, not automatic bug classification.",
        ],
        runs=[analyze_run(run) for run in source["runs"]],
    )
    output = args.output or args.input.with_name("dispatch_analysis.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k not in ("local_pairs", "groups")} for r in report["runs"]]
        )
    )


if __name__ == "__main__":
    main()
