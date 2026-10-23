# SPDX-License-Identifier: Apache-2.0
"""只更新采用v32/v34后的MPMD两列；复用其余合格记录，不启动设备任务。"""

import copy
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COLUMNS = ["Native", "BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD"]


def read(path):
    return json.loads(path.read_text())


def main():
    base_path = HERE.parent / "csa_performance_20261023/PERFORMANCE.json"
    base = read(base_path)
    qualified = read(HERE / "scheduling_v32_v34_qualified.json")
    adoption = read(HERE / "scheduling_v36_adoption.json")
    assert qualified["adopted"] and adoption["adopted"]
    cases = read(HERE.parent / "csa_performance_20261022_aicpu/cases.json")["cases"]
    cells = copy.deepcopy(base["cells"])
    updates = {}
    for case_id, variant in (("C02", "v34"), ("C05", "v32")):
        case = next(c for c in cases if c["id"] == case_id)
        gate = qualified["variants"][variant]
        pair_path = HERE / f"runs/aicpu_{variant}_pair_{case['name']}/result.json"
        full_path = Path(gate["full_source"])
        reports = [read(pair_path), read(full_path)]
        pair_task = read(HERE / f"scheduling_{variant}_preparation.json")["task_id"]
        full_task = gate["task_bindings"][1]["task_id"]
        assert all(d["passed"] and d["stage"] == "complete" and d["case"] == case for d in reports)
        samples, raws = [], []
        for report in reports:
            windows = [w for w in report["windows"] if w["side"] == "mpmd"]
            assert len(windows) == 10
            for window in windows:
                raw = read(Path(window["raw"]))
                rows = raw["aicore_tasks"]
                assert raw["chip_swimlane_level"] == 1 and all(r[3] > 0 and r[4] >= r[3] for r in rows)
                value = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
                assert abs(value - window["incore_span_us"]) < 1e-8
                samples.append(value)
                raws.append(window["raw"])
        outer = reports[1]["outer_profile"]
        assert outer["available"] and outer["span_includes_all_streams"] and len(outer["span_us_all"]) == 30
        assert Path(outer["raw_csv"]).exists()
        common = dict(
            status="PASS",
            reuse=True,
            selected_implementation="v36: v22 plus bounded output rectangle and physical-plan capacity",
            measured_component=variant,
            production_commit=adoption["production_commit"],
            input_contract=case,
            source_date="2026-10-23",
            shared_toolchain=adoption["shared_toolchain"],
            weight_and_initial_state_source=str(pair_path),
            devices_by_source={str(pair_path): reports[0]["device_id"], str(full_path): reports[1]["device_id"]},
            framework_and_cann_evidence="scheduling_v31_v34_task_evidence.json",
            correctness=(
                "eager/graph/post-profile plus 10 same-address cases and 4 continuous steps: bitwise/guards PASS"
            ),
            qualification="scheduling_v32_v34_qualified.json",
            implementation_equivalence="scheduling_v36_codegen_equivalence.json",
            source_reports=[str(pair_path), str(full_path)],
        )
        updates[case_id] = {
            "TND-MPMD-L1": dict(
                common,
                samples_us=samples,
                valid_windows=len(samples),
                median_us=statistics.median(samples),
                raw=raws,
                source_tasks=[pair_task, full_task],
                sample_definition="one full CSA graph replay, first incore begin to last incore end including gaps",
            ),
            "TND-MPMD-AICPU": dict(
                common,
                samples_us=outer["span_us_all"],
                valid_windows=30,
                median_us=statistics.median(outer["span_us_all"]),
                raw=[outer["raw_csv"]],
                source_reports=[str(full_path)],
                source_tasks=[full_task],
                sample_definition="full CSA all-device-stream span, includes AICPU; benchmark reset excluded",
                l1_outer_pairing=outer["paired_l1_note"],
            ),
        }
        cells[case_id].update(updates[case_id])
    base_lines = (HERE.parent / "csa_performance_20261023/PERFORMANCE.md").read_text().splitlines()
    labels = [line.split("|")[1].strip() for line in base_lines[2:]]
    assert len(labels) == len(cases) == 8
    lines = base_lines[:2]
    medians = {}
    for case, label in zip(cases, labels):
        row = cells[case["id"]]
        values = []
        for column in COLUMNS:
            cell = row[column]
            if cell["status"] == "N/A":
                values.append("N/A")
                continue
            assert cell["status"] == "PASS" and len(cell["samples_us"]) >= 10
            median = statistics.median(cell["samples_us"])
            assert abs(median - cell["median_us"]) < 1e-8
            values.append(f"{median:.2f}")
        percentage = 100 * (row["Native"]["median_us"] - row["TND-SPMD"]["median_us"]) / row["TND-SPMD"]["median_us"]
        lines.append("| " + " | ".join([label, *values, f"{percentage:.2f}%"]) + " |")
        medians[case["id"]] = {column: row[column].get("median_us") for column in COLUMNS}
    (HERE / "scheduling_v36_PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    record = dict(
        base_report=str(base_path.relative_to(REPO)),
        unchanged_cells="All cells except C02/C05 TND-MPMD columns inherit the base report, including raw samples.",
        reference_columns="Native/BSH/SPMD remain the explicitly selected validated snapshots in the base report.",
        mpmd_off_domain="Six other cases retain v22 arithmetic/launch paths; new tiers do not activate.",
        updates=updates,
        resolved_medians_us=medians,
        sample_unit="one replay, median of all valid windows; not window averages or fastest-sample selection",
        original_goal_recomputed=False,
        adoption="scheduling_v36_adoption.json",
    )
    (HERE / "scheduling_v36_PERFORMANCE.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("8 rows regenerated; only four changed cells updated, no new NPU run")


if __name__ == "__main__":
    main()
