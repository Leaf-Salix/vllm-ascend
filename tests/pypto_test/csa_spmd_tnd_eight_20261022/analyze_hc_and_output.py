# SPDX-License-Identifier: Apache-2.0
"""Record limits of HC phase attribution and observed O_A full/tail intervals."""

import json
import statistics
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    old, new = [
        json.loads((here / name).read_text())
        for name in ("retained_C04_phase_sdk_v1.json", "hc_cooperative_v4_C04_phase.json")
    ]
    measurement = json.loads((here / "hc_cooperative_v4_results.json").read_text())["cases"]["C04"]
    common = [
        "META",
        "PARTIAL",
        "RAW",
        "GATES",
        "NORM",
        "COMB",
        "QA",
        "QR",
        "QUERY",
        "SCORE_VECTOR",
        "ATTN_VECTOR",
        "OB",
        "OUT",
    ]
    phases = {}
    for phase in common:
        before, after = [
            [window["phases"][phase]["latest_end_us"] for window in data["windows"]] for data in (old, new)
        ]
        phases[phase] = dict(
            retained=before,
            candidate=after,
            retained_median=statistics.median(before),
            candidate_median=statistics.median(after),
        )
    report = dict(
        performance_eligible=False,
        source_reports=["retained_C04_phase_sdk_v1.json", "hc_cooperative_v4_C04_phase.json"],
        common_payload_endpoints_us=phases,
        excluded_phase=dict(
            INPUT="Candidate publishes FP32/RMS chunks before inverse combine; old INPUT included inverse"
        ),
        complete_L1_us=dict(retained=measurement["retained_median_us"], candidate=measurement["median_us"]),
        direction_disagrees_with_complete_L1=True,
        decision="Keep the complete-L1 rejection; cross-run instrumented endpoint differences "
        "do not establish the regression cause",
        limitations=[
            "Instrumentation and separate executions can change scheduling and timing",
            "No endpoint differences are claimed as exclusive service cost or removable latency",
        ],
    )
    assert phases["OUT"]["candidate_median"] < phases["OUT"]["retained_median"]
    assert measurement["median_us"] > measurement["retained_median_us"]
    with (here / "hc_cooperative_v4_C04_phase_comparison.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")

    diagnostic = json.loads((here / "retained_C05_phase_sdk_v1.json").read_text())
    windows = []
    for window in diagnostic["windows"]:
        raw = json.loads(Path(window["raw"]).read_text())
        rows = [row for row in raw if row.get("name", "").startswith("OA_GROUP_")]
        samples = dict(full=[], tail=[])
        for core in range(24):
            owned = sorted((row for row in rows if row["core"] == core), key=lambda row: row["notify_end_us"])
            for row in owned[1:]:
                group = int(row["name"].split("_")[-1])
                jobs = [logical for logical in range(group * 8, (group + 1) * 8) if logical % 24 == core]
                assert len(jobs) == 1 and abs(row["wait_us"]) < 1e-9
                kind = "full" if jobs[0] % 8 // 4 == 0 else "tail"
                samples[kind].append(dict(core=core, group=group, interval_us=row["interval_us"]))
        assert len(samples["full"]) == len(samples["tail"]) == 20
        windows.append(
            dict(
                raw=window["raw"],
                samples=samples,
                medians_us={
                    kind: statistics.median(row["interval_us"] for row in values) for kind, values in samples.items()
                },
            )
        )
    output = dict(
        performance_eligible=False,
        source_report="retained_C05_phase_sdk_v1.json",
        windows=windows,
        fixed_source_geometry=dict(tokens=144, row_tile=128, column_tile=256, groups=8, AICs=24),
        selected_intervals="Only the second/third owned O_A group; "
        "previous publication is an O_A group and GM wait is zero",
        limitations=[
            "Intervals still include local loops, DMA, pipeline and publication, not pure matrix time",
            "Do not fit candidate N192 latency by scaling these N256 observations",
        ],
        rectangle_source_budget=dict(
            jobs=48,
            jobs_per_AIC=2,
            full_column_jobs=40,
            tail_column_jobs=8,
            row_capacity=160,
            full_width=192,
            tail_width=64,
            max_accumulator_bytes=160 * 192 * 4,
            logical_weight_MiB=dict(original=128, rectangle=64),
            logical_activation_MiB=dict(original=36, rectangle=54),
        ),
    )
    with (here / "C05_output_observed_service.json").open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print("HC endpoint direction differs from formal L1; no causal attribution")
    print("O_A full/tail per-window medians", [window["medians_us"] for window in windows])


if __name__ == "__main__":
    main()
