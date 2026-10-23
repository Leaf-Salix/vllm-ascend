# SPDX-License-Identifier: Apache-2.0
"""Adopt only after production CPP/ABI match every validated case."""

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analyze_priority_phase import executable


def write_new(path, data):
    with path.open("x") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")


def main():
    here = Path(__file__).resolve().parent
    source = json.loads((here / "output_rectangle_production_v1.json").read_text())
    root = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")} == source["files"]
    for name in ("output_rectangle_v3_codegen_validation.json", "output_rectangle_v3_runtime_validation.json"):
        assert json.loads((here / name).read_text())["passed"]
    rows = {}
    for number in range(1, 9):
        case = f"C{number:02d}"
        build = here / f"runs/compile_output_rectangle_production_{case}_v1"
        assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
        candidate, validated = build / "kernel", here / f"runs/compile_output_rectangle_{case}_v3/kernel"
        abi = lambda p: json.loads((p / "compiled_meta.json").read_text())["kernel_abi"]
        assert abi(candidate) == abi(validated)
        paths = sorted(p.relative_to(candidate) for p in candidate.rglob("*.cpp"))
        assert len(paths) == 4
        assert paths == sorted(p.relative_to(validated) for p in validated.rglob("*.cpp"))
        hashes = {}
        for path in paths:
            actual, expected = [executable((p / path).read_text()) for p in (candidate, validated)]
            assert actual == expected, (case, path)
            hashes[str(path)] = hashlib.sha256(actual.encode()).hexdigest()
        rows[case] = dict(
            candidate=str(candidate),
            validated=str(validated),
            all_four_cpp_equal=True,
            kernel_abi_equal=True,
            executable_cpp_sha256=hashes,
        )
    codegen = dict(
        passed=True,
        cases=rows,
        source="output_rectangle_production_v1.json",
        source_contract_tests=38,
        all_eight_candidate_codegen="output_rectangle_v3_codegen_validation.json",
        device_execution=False,
        performance_eligible=False,
        comparison="Only whole // pto: source-location comment lines removed",
    )
    pilot = json.loads((here / "output_rectangle_v3_results.json").read_text())["cases"]["C05"]
    retained = json.loads((here / "retained_physical_capacity.json").read_text())
    assert pilot["pilot_positive"] and pilot["strict_checks_and_guards"]
    assert pilot["frozen_denominator_us"] == retained["cases"]["C05"]["baseline_median_us"]
    retained["cases"]["C05"].update(
        current_median_us=pilot["median_us"],
        ratio=pilot["ratio_to_original"],
        measurement_reused=True,
        measurement="output_rectangle_v3_results.json",
        all_samples_us=pilot["all_samples_us"],
    )
    ratio = math.exp(sum(math.log(row["ratio"]) for row in retained["cases"].values()) / 8)
    retained.update(
        candidate="output_rectangle_production_v1",
        source_integration="output_rectangle_integration.json",
        compiled_equivalence="output_rectangle_production_v1_codegen.json",
        equal_weight_geometric_mean_ratio=ratio,
        goal_achieved=ratio <= retained["target_ratio"],
        cpu_contract_tests=38,
    )
    retained["retained_mechanisms"].append("M160/N192 with a static N64 tail and original O_B owners (C05)")
    retained["evidence"] += [
        "retained_physical_capacity.json",
        "output_rectangle_v3_results.json",
        "output_rectangle_v3_codegen_validation.json",
        "output_rectangle_v3_runtime_validation.json",
        "output_rectangle_production_v1_codegen.json",
        "output_rectangle_integration.json",
    ]
    integration = dict(
        candidate="output_rectangle_production_v1",
        adopted=True,
        source_manifest="output_rectangle_production_v1.json",
        formatted_production_codegen="output_rectangle_production_v1_codegen.json",
        performance="output_rectangle_v3_results.json",
        device_validation="output_rectangle_v3_runtime_validation.json",
        production_contract_tests=38,
        original_ND_helper_unchanged=True,
        push_authorized=False,
    )
    write_new(here / "output_rectangle_production_v1_codegen.json", codegen)
    write_new(here / "output_rectangle_integration.json", integration)
    write_new(here / "retained_output_rectangle.json", retained)
    source.update(adopted=True, production_codegen_pending=False)
    (here / "output_rectangle_production_v1.json").write_text(json.dumps(source, indent=2) + "\n")
    state_path = here / "current_state.json"
    state = json.loads(state_path.read_text())
    state.update(
        updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        pending_cpu=None,
        pending_tasks={},
        current_implementation="output_rectangle_production_v1",
        retained_results="retained_output_rectangle.json",
        production_source_manifest="output_rectangle_production_v1.json",
        formal_geometric_mean_ratio=ratio,
        latest_stage="output_rectangle_adopted_all_eight_production_cpp_equal",
    )
    state["accepted_candidates"].append(
        dict(candidate="output_rectangle_production_v1", case_id="C05", evidence="retained_output_rectangle.json")
    )
    state["prepared_priority_candidates"]["output_rectangle_v3"].update(
        status="adopted_as_output_rectangle_production_v1",
        adopted=True,
        integration="output_rectangle_integration.json",
    )
    state["promising_candidates"] = [
        row for row in state["promising_candidates"] if row["candidate"] != "output_rectangle_v3"
    ]
    state["next_steps"][0] = (
        "Preserve five retained mechanisms and original eight-case denominator; 10% goal remains active"
    )
    state["next_steps"][1] = (
        "Model a C04 output-group ownership grid that reduces activation reloads and balances columns per AIC; "
        "check resource and dependency costs before device work"
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(dict(C05_us=pilot["median_us"], geometric_mean_ratio=ratio, reduction_pct=100 * (1 - ratio))))


if __name__ == "__main__":
    main()
