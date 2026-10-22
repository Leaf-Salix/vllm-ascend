# SPDX-License-Identifier: Apache-2.0
"""Read existing eight-case measurements; never compile or execute a device job."""

import csv
import hashlib
import json
import math
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo


def read_json(path):
    return json.loads(path.read_text())


def checked_median(samples, count):
    assert len(samples) == count and all(math.isfinite(value) and value > 0 for value in samples)
    result = median(samples)
    ordered = sorted(Decimal(str(value)) for value in samples)
    independent = (ordered[(count - 1) // 2] + ordered[count // 2]) / 2
    assert abs(Decimal(str(result)) - independent) < Decimal("0.000000001")
    return result


def rounded(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    workspace = here.parents[4]
    other_repo = workspace / "vllm-ascend-dsv4-pto-0251rc1"
    native_root = repo / "tests/pypto_test/csa_five_way_latest_20261020"
    mpmd_root = other_repo / "tests/pypto_test/csa_multitask_backport_20261020"
    matrix_name = "MATRIX_qa_shape_gate_v1_shape_gate_v1.json"
    matrix = read_json(mpmd_root / matrix_name)
    assert (repo / "tests/pypto_test/csa_multitask_backport_20261020" / matrix_name).read_bytes() == (
        mpmd_root / matrix_name
    ).read_bytes()
    assert matrix["passed"] and read_json(mpmd_root / "QA_SHAPE_GATE_STATUS.json")["production_adopted"]
    cohort_path = here / "cohort_output_balance_v4_result.json"
    cohort = read_json(cohort_path)
    assert read_json(here / "latest_complete_cohort.json")["report"] == cohort_path.name
    cases = [(131072, batch) for batch in (1, 4, 8, 16, 24, 28)] + [(8192, batch) for batch in (16, 32)]
    mpmd_rows = {(row["history"], row["batch"]): row for row in matrix["rows"]}
    tnd_rows = {(row["history"], row["batch"]): row for row in cohort["cases"]}
    assert len(matrix["rows"]) == len(cohort["cases"]) == 8
    assert set(mpmd_rows) == set(tnd_rows) == set(cases)
    jobs = {row["task_id"]: row for row in read_json(here / "jobs.json")}
    old_native = {}
    with (native_root / "PERFORMANCE.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            old_native[int(row["history"]), int(row["batch"])] = float(row["native_us"])
    old_native[131072, 1] = read_json(repo / "tests/pypto_test/csa_native_b1_20261021/PERFORMANCE.json")["native_us"]
    result = []
    for history, batch in cases:
        attempt = "native_supplement_20261021_v1" if batch == 1 else "native_v1"
        native_path = native_root / "results" / f"h{history}_b{batch}" / attempt / "report.json"
        native = read_json(native_path)
        assert (native["history"], native["batch"], native["query_tokens"], native["layer_index"]) == (
            history, batch, 6, 4
        )
        assert native["status"] == "MEASURED" and set(native["native_weight_formats"].values()) == {29}
        profile = native["device_profile"]
        assert profile["available"] and profile["span_includes_all_streams"] and profile["replays"] == 30
        assert native["compiler"]["wrapper_compiled"] and native["compiler"]["installed_static_packages"] > 0
        assert native["multistream"]["dsa_overlap"]
        assert all(call["status"] == "RETURNED_SUCCESSFULLY" for call in native["super_kernel_graph_calls"])
        native_samples = profile["span_us_all"]
        native_us = checked_median(native_samples, 30)
        assert native_us == old_native[history, batch] == native["device_span_us"]

        mpmd_row = mpmd_rows[history, batch]
        mpmd_path = mpmd_root / mpmd_row["result"]
        mpmd = read_json(mpmd_path)
        assert (mpmd["history"], mpmd["batch"], mpmd["query_tokens"]) == (history, batch, 6)
        assert mpmd["passed"] and mpmd["performance_measured"] and mpmd["device_execution"]
        mpmd_windows = [row for row in mpmd["l1"]["windows"] if row["side"] == "multitask_candidate"]
        assert all(row["cores"] == 72 and row["clock_freq_hz"] == 50000000 for row in mpmd_windows)
        mpmd_l1_samples = [row["incore_span_us"] for row in mpmd_windows]
        assert mpmd_l1_samples == mpmd_row["l1_us"]["multitask_candidate"]["samples_us"]
        mpmd_l1_us = checked_median(mpmd_l1_samples, 5)
        assert mpmd_l1_us == mpmd_row["l1_us"]["multitask_candidate"]["median_us"]
        outer = mpmd["l1"]["outer"]["statistics_us"]["multitask_candidate"]["samples_us"]
        assert outer == mpmd_row["outer"]["statistics_us"]["multitask_candidate"]["samples_us"]
        mpmd_outer_us = checked_median(outer, 5)

        tnd_row = tnd_rows[history, batch]
        job = jobs[tnd_row["task_id"]]
        tnd_path = Path(job["output"]) / "result.json"
        tnd = read_json(tnd_path)
        assert (tnd["history"], tnd["batch"], tnd["query_tokens"]) == (history, batch, 6)
        assert job["last_observed_status"] == "completed (exit=0)"
        assert tnd["passed"] and tnd["functional_passed"] and tnd["performance_measured"]
        assert tnd["original_spmd_attribution"]["candidate_exactly_matches"]
        values, samples = {}, {}
        for side, field in (("spmd_original", "bsh_us"), ("spmd_tnd", "tnd_us")):
            windows = [row for row in tnd["l1"]["windows"] if row["side"] == side]
            assert all(row["cores"] == 72 and row["clock_freq_hz"] == 50000000
                       and row["task_count"] == 1 for row in windows)
            samples[side] = [row["incore_span_us"] for row in windows]
            values[side] = checked_median(samples[side], 5)
            assert values[side] == tnd_row[field]
        percentage = (native_us - values["spmd_tnd"]) / values["spmd_tnd"] * 100
        independent_percentage = (Decimal(str(native_us)) / Decimal(str(values["spmd_tnd"])) - 1) * 100
        assert abs(percentage - float(independent_percentage)) < 1e-10
        paths = [native_path, mpmd_path, tnd_path]
        result.append({
            "history": history, "batch": batch, "query_tokens": 6,
            "native_us": native_us, "bsh_mpmd_outer_us": mpmd_outer_us, "bsh_mpmd_l1_us": mpmd_l1_us,
            "bsh_spmd_l1_us": values["spmd_original"], "tnd_spmd_l1_us": values["spmd_tnd"],
            "native_minus_tnd_over_tnd_pct": percentage,
            "samples": {"native": native_samples, "mpmd_outer": outer,
                        "mpmd_l1": mpmd_l1_samples, **samples},
            "source_files": [str(path) for path in paths],
            "source_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
            "tnd_task_id": tnd_row["task_id"],
        })
    notes = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "new_device_execution": False, "statistic": "median", "latency_unit": "us",
        "formula": "(native_us - tnd_spmd_l1_us) / tnd_spmd_l1_us * 100",
        "scope": "8 S6 cases, TP1/single card, W8A8/NZ2/layer4; HC_pre+RMSNorm+CSA+HC_post",
        "native": "Oct20 seven-case30-sample Profiler full-device/multistream span; Oct21 B1 supplement",
        "mpmd": "Oct21 accepted qa_shape_gate_v1/a658188c; candidate role is now adopted; both columns same version",
        "mpmd_outer": "5 NPU Event samples around graph.replay, includes AICPU/runtime; not Profiler span",
        "mpmd_l1": "5 existing full-chain L1 earliest-incore to latest-end windows",
        "bsh_spmd_and_tnd": (
            "Paired v4 complete cohort; a241fa61+static B1 shape fix/native150 vs adopted output balance"
        ),
        "caveats": ["Five columns were not measured together; source revisions and toolchains differ",
                    "Native/MPMD outer are outer scope; remaining three columns are incore L1 scope",
                    "Outer minus L1 is not a measured pure AICPU cost",
                    "Percentage uses TND denominator and unrounded inputs; not same-scope end-to-end acceleration",
                    "Unadopted QA/WS/Score-owner TND candidates are excluded"],
        "rows": result,
    }
    output = workspace / "downloads/csa_eight_cases_five_paths_20261021"
    output.mkdir(exist_ok=True)
    (output / "sources_and_samples.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    (here / "five_way_existing_20261021.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    fields = ["history", "batch", "query_tokens", "native_us", "bsh_mpmd_outer_us", "bsh_mpmd_l1_us",
              "bsh_spmd_l1_us", "tnd_spmd_l1_us", "native_minus_tnd_over_tnd_pct"]
    with (output / "performance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result:
            writer.writerow({key: row[key] if key in fields[:3] else rounded(row[key]) for key in fields})
    print("| 场景（S6） | Native | BSH MPMD 含AICPU | BSH MPMD L1 | BSH SPMD L1 | TND SPMD L1 | (Native−TND)/TND |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in result:
        label = f"{row['history'] // 1024}K / B{row['batch']}"
        cells = [rounded(row[key]) for key in fields[3:]]
        cells[-1] += "%"
        print("| " + " | ".join([label, *cells]) + " |")
    print("Saved:", output)


if __name__ == "__main__":
    main()
