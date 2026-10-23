# SPDX-License-Identifier: Apache-2.0
"""Compare the C02 capacity pilot with its qualified same-device historical baseline."""

import json
import math
import statistics
from pathlib import Path

from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def samples_from_windows(windows, side):
    samples = []
    for window in windows:
        raw = json.loads(Path(window["raw"]).read_text())
        rows, hz = raw["aicore_tasks"], raw["metadata"]["clock_freq_hz"]
        assert raw["chip_swimlane_level"] == 1 and hz > 0 and window["side"] == side
        assert len(rows) == len({row[0] for row in rows}) == 72 and len({row[1] for row in rows}) == 1
        assert all(len(row) == 6 and row[4] >= row[3] > 0 for row in rows)
        value = (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / hz
        assert math.isclose(value, window["incore_span_us"], rel_tol=0, abs_tol=1e-8)
        samples.append(value)
    assert len(samples) >= 10
    return samples


def main():
    here = Path(__file__).resolve().parent
    run = here / "runs/h262144_b16_s6_physical_capacity_2080_v1"
    data = json.loads((run / "result.json").read_text())
    reference = json.loads((here / "C02_reused_reference.json").read_text())
    source = json.loads((here / "physical_capacity_2080_v1.json").read_text())
    assert reference["passed"] and reference["baseline_algorithm_unchanged_for_this_case"]
    require_same_sdk(data["shared_toolchain"], reference["shared_toolchain"])
    assert data["passed"] and data["device_execution"] and data["stage"] == "complete"
    assert data["case"] == reference["case"] and data["device_id"] == reference["device_id"] == 9
    assert data["source_hashes"]["candidate"] == source["files"]
    assert set(data["checks"]) == {
        f"{side}_{stage}" for side in ("reference", "candidate") for stage in ("eager", "graph", "post_l1")
    }
    for check in data["checks"].values():
        assert all(check["guards"].values()) and all(row["bit_equal"] for row in check["numerical"].values())
    samples = samples_from_windows(data["windows"], "candidate")
    baseline_samples = samples_from_windows(reference["windows"], "spmd")
    assert samples == data["statistics_us"]["candidate"]["samples"]
    assert baseline_samples == reference["samples_us"]
    median, baseline = statistics.median(samples), statistics.median(baseline_samples)
    assert baseline == reference["median_us"]
    cpu = here / "runs/compile_physical_capacity_C02_v1/kernel/ptoas/dynamic_csa_full.cpp"
    matching = [
        str(p)
        for p in (run / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
        if executable(p.read_text()) == executable(cpu.read_text())
    ]
    assert matching
    witness = data["physical_plan_observation"]
    assert witness["observation_after_all_L1_and_strict_checks"]
    assert witness["physical_indices_match_independent_host_mapping"]
    assert witness["physical_representation_differs_from_logical_indices"]
    assert witness["actual_table_columns"] == 2050 and witness["candidate_capacity"] == 2080
    report = dict(
        candidate="physical_capacity_2080_v1",
        case_id="C02",
        adopted=False,
        task_id="task_20261023_093314_284037929851",
        source_report=str(run / "result.json"),
        reference_certificate="C02_reused_reference.json",
        reference_remeasured=False,
        all_samples_us=samples,
        baseline_samples_us=baseline_samples,
        raw_windows=[window["raw"] for window in data["windows"]],
        median_us=median,
        baseline_median_us=baseline,
        ratio_to_frozen=median / baseline,
        change_vs_retained_pct=100 * (median / baseline - 1),
        strict_numerical_and_guards=True,
        measured_cpp_matches_cpu_gate=matching,
        path_witness=witness,
        pilot_positive=median < baseline,
        limitation="Pilot only; no adoption or eight-case gain update without tier integration and state validation",
    )
    with (here / "physical_capacity_2080_v1_C02_result.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("median_us", "baseline_median_us", "change_vs_retained_pct")}))


if __name__ == "__main__":
    main()
