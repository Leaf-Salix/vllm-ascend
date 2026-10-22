# SPDX-License-Identifier: Apache-2.0
"""Summarize existing evidence and output work geometry without executing NPU code.

Logical FLOPs/load bytes are source-level counts, not measured pipe utilization
or HBM traffic. Diagnostic endpoints are not exclusive stage durations.
"""

import argparse
import json
import math
import runpy
import statistics
from pathlib import Path

CORES = 24
OUTPUT_GROUPS = 8
OUTPUT_K = 4096
OUTPUT_N = 1024


def output_work(tokens, row_tile, column_tile):
    row_blocks = math.ceil(tokens / row_tile)
    column_blocks = OUTPUT_N // column_tile
    owners = [dict(core=core, jobs=0, valid_mn=0, aligned_mn=0, weight_load_bytes=0) for core in range(CORES)]
    for logical in range(OUTPUT_GROUPS * row_blocks * column_blocks):
        row_block = (logical % (row_blocks * column_blocks)) // column_blocks
        valid = min(row_tile, tokens - row_block * row_tile)
        owner = owners[logical % CORES]
        owner["jobs"] += 1
        owner["valid_mn"] += valid * column_tile
        owner["aligned_mn"] += math.ceil(valid / 16) * 16 * column_tile
        owner["weight_load_bytes"] += OUTPUT_K * column_tile * 2
    assert sum(owner["valid_mn"] for owner in owners) == OUTPUT_GROUPS * tokens * OUTPUT_N
    quantities = {}
    for key in ("jobs", "valid_mn", "aligned_mn", "weight_load_bytes"):
        values = [owner[key] for owner in owners]
        quantities[key] = dict(
            total=sum(values),
            maximum=max(values),
            mean=statistics.mean(values),
            maximum_over_mean=max(values) / statistics.mean(values),
        )
    return dict(
        row_tile=row_tile,
        column_tile=column_tile,
        owners=owners,
        quantities=quantities,
        useful_flops=2 * OUTPUT_K * quantities["valid_mn"]["total"],
        unique_weight_bytes=OUTPUT_GROUPS * OUTPUT_K * OUTPUT_N * 2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    production = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    select = runpy.run_path(str(production / "policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    score_model = json.loads((here / "score_owner_work_model.json").read_text())
    phase = json.loads((here / "baseline_C05_phase_sdk_v1.json").read_text())
    report = dict(
        device_execution=False,
        performance_eligible=False,
        source_policy=str(production / "policy.py"),
        source_output=str(production / "output_projection_a_stages.py"),
        goal_budget=dict(
            equal_weight_cases=8,
            target_geomean_ratio=0.9,
            ratio_if_only_two_cases_improve_ten_percent=0.9 ** (2 / 8),
            required_ratio_per_case_if_only_two_equal_improvements=0.9 ** (8 / 2),
            C04_original_us=778.39,
            C04_retained_us=763.65,
            C04_additional_us_to_ten_percent_of_original=763.65 - 778.39 * 0.9,
            C05_original_us=1014.46,
            C05_us_to_ten_percent_of_original=1014.46 * 0.1,
            per_case_ten_percent_is_illustrative_not_an_added_acceptance_rule=True,
        ),
        cases={},
        diagnostic_source="baseline_C05_phase_sdk_v1.json",
        C05_phase_endpoints={},
        limitations=[
            "O_A owner model follows existing (group,row_block,column_block) % 24 geometry",
            "M16 alignment is an arithmetic-work proxy, not measured Cube cycles",
            "Weight load bytes count logical GM-to-L1 requests, not measured HBM traffic",
            "Hypothetical single-M/N128 geometry has no compile/resource/device validation",
            "Score rows-times-steps exclude prefix arrival, Scalar, memory and FIFO costs",
            "Three instrumented windows are diagnostic only; endpoints/spreads cannot be added as savings",
            "Current C04 retained implementation has no matched task-level timing decomposition here",
            "Historical failed wave data use their original version and are not current performance baselines",
        ],
    )
    for case in cases:
        policy = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        data = dict(
            tokens=case["valid_tokens"],
            output_work_balance=policy["OUTPUT_WORK_BALANCE"],
            current_oa=output_work(case["valid_tokens"], 128, policy["OA_COLS"]),
        )
        score = score_model["cases"][case["id"]]
        if "original" in score:
            data["score_work_model"] = dict(
                maximum_over_mean=score["original"]["max_over_mean"],
                distinct_steps=sorted({core["score_steps"] for core in score["original"]["cores"]}),
            )
        if 128 < case["valid_tokens"] <= 192:
            data["unimplemented_single_m_n128_geometry"] = output_work(case["valid_tokens"], 192, 128)
        report["cases"][case["id"]] = data
    for name in (
        "QB",
        "KV_PROJECT",
        "SCORE_CUBE",
        "SCORE_VECTOR",
        "TOPK",
        "ATTN_PLAN",
        "ATTN_CUBE",
        "ATTN_VECTOR",
        "OA_GROUP_7",
        "OB",
        "OUT",
    ):
        rows = [window["phases"][name] for window in phase["windows"]]
        report["C05_phase_endpoints"][name] = {
            key: statistics.median(row[key] for row in rows) for key in ("latest_end_us", "completion_spread_us")
        }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    for case_id, data in report["cases"].items():
        work = data["current_oa"]["quantities"]
        print(
            case_id,
            "O_A max/mean",
            round(work["valid_mn"]["maximum_over_mean"], 4),
            "logical weight MiB",
            work["weight_load_bytes"]["total"] / (1024**2),
        )


if __name__ == "__main__":
    main()
