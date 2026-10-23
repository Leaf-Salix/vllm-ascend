# SPDX-License-Identifier: Apache-2.0
"""Recompute scope and scheduling facts without new device measurements."""

import json
import statistics
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    read = lambda name: json.loads((here / name).read_text())
    retained = read("retained_merged_schedule.json")
    ratio = statistics.geometric_mean(case["ratio"] for case in retained["cases"].values())
    assert abs(ratio - retained["equal_weight_geometric_mean_ratio"]) < 1e-12
    groups = 24
    attention = {}
    for case in retained["cases"]:
        descriptor = read(f"descriptors/{case}_merged_schedule_v1.json")
        tokens = descriptor["buffers"]["rope_cos"]["shape"][0]
        jobs = [len(range(core, tokens, groups)) for core in range(groups)]
        assert sum(jobs) == tokens
        attention[case] = dict(
            tokens=tokens,
            query_count_by_cluster=jobs,
            active_clusters=sum(count > 0 for count in jobs),
            minimum_queries=min(jobs),
            maximum_queries=max(jobs),
            maximum_over_all_cluster_mean=max(jobs) / (tokens / groups),
            interpretation="Capacity-query assignment only; not a measured utilization or a service-cost estimate",
        )
    phases = {}
    for case in ("C04", "C05"):
        trace = read(f"retained_{case}_phase_sdk_v1.json")
        phases[case] = {
            name: {
                key: statistics.median(window["phases"][name][key] for window in trace["windows"])
                for key in ("latest_end_us", "completion_spread_us")
            }
            for name in ("QUERY", "SCORE_CUBE", "SCORE_VECTOR", "TOPK", "ATTN_PLAN", "ATTN_VECTOR", "OB", "OUT")
        }
    readiness = read("score_readiness_model.json")
    leaf_ready = {
        case: {
            "whole_owner_latest_step": max(data["whole_owner_root_ready"]),
            "per_leaf_earliest_step": min(data["hypothetical_per_leaf_root_ready"]),
            "first_query_ready_with_all_three_dependency_changes": min(
                data["per_leaf_and_per_query_topk_and_first_attention_query_only"]["attention_ready"]
            ),
        }
        for case, data in readiness["cases"].items()
    }
    locality = read("retained_C05_kv_access.json")
    report = dict(
        new_device_execution=False,
        performance_eligible=False,
        production_candidate="merged_schedule_v1",
        unchanged_goal_denominator=retained["baseline_kind"],
        retained_geometric_mean_ratio=ratio,
        target_geometric_mean_ratio=0.9,
        required_equal_reduction_relative_to_retained_pct={
            str(affected): 100 * (1 - (0.9 / ratio) ** (8 / affected)) for affected in (2, 4, 5, 8)
        },
        attention_cluster_geometry=attention,
        instrumented_phase_frontiers=phases,
        idealized_score_ready_steps=leaf_ready,
        C05_measured_locality={
            key: locality[key]
            for key in (
                "valid_compressed_rows",
                "contiguous_runs_within_existing_64_row_lane_segments",
                "upper_bound_row_request_reduction_fraction",
                "whole_page_overfetch_ratio_per_existing_lane_segment",
                "consecutive_query_row_set_reuse",
                "exact_64_segments",
                "exact_128_blocks",
            )
        },
        actual_prefetch_mechanism_audit="attention_prefetch_v1_codegen.json",
        source_reports=[
            "retained_merged_schedule.json",
            "retained_C04_phase_sdk_v1.json",
            "retained_C05_phase_sdk_v1.json",
            "score_readiness_model.json",
            "retained_C05_kv_access.json",
            "attention_interface_traffic.json",
            "retained_dependency_opportunities.json",
        ],
        limitations=[
            "No stage frontier, core completion spread or sum of pipe times is asserted to be removable latency",
            "Idealized readiness excludes resource occupation, sort/plan cost, state restoration and memory contention",
            "Capacity query counts do not measure phase occupancy, valid sparse blocks or runtime mixed-query cost",
            "The gain budget assumes other case medians unchanged and equal relative changes on affected cases",
            "Historical BSH negative experiments constrain design but are not same-SDK performance baselines",
        ],
    )
    with (here / "structural_priorities.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "required_reduction_pct": report["required_equal_reduction_relative_to_retained_pct"],
                "attention_active_clusters": {case: value["active_clusters"] for case, value in attention.items()},
                "score_ready_steps": leaf_ready,
            }
        )
    )


if __name__ == "__main__":
    main()
