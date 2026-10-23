# SPDX-License-Identifier: Apache-2.0
"""Bind the positive rectangle pilot to eight-case CPU and runtime gates."""

import json
import runpy
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    source = json.loads((here / "output_rectangle_v3.json").read_text())
    select = runpy.run_path(str(Path(source["source"]) / "policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    builds = {}
    for case in cases:
        case_id = case["id"]
        old = json.loads((here / f"descriptors/{case_id}_physical_capacity_tier_v1.json").read_text())
        selected = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=6,
            max_seq_len=max(case["seq_lens"]),
            compressed_table_columns=old["buffers"]["compressed_table"]["shape"][1],
        ).constexprs()
        expected = dict(old["policy"], OA_ROWS=128)
        if case_id == "C05":
            expected.update(OA_ROWS=160, OA_COLS=192)
        assert selected == expected, case_id
        old["policy"] = selected
        descriptor = here / f"descriptors/{case_id}_output_rectangle_v1.json"
        if descriptor.exists():
            assert json.loads(descriptor.read_text()) == old
        else:
            with descriptor.open("x") as out:
                out.write(json.dumps(old, indent=2) + "\n")
        builds[case_id] = dict(
            descriptor=str(descriptor.relative_to(here)),
            candidate=f"runs/compile_output_rectangle_{case_id}_v3/kernel",
            retained=f"runs/compile_physical_capacity_tier_{case_id}_v1/kernel",
            cpu_already_passed=case_id in ("C04", "C05"),
            requires_cpp_identity_to_retained=case_id != "C05",
        )
    report = dict(
        source="output_rectangle_v3.json",
        performance="output_rectangle_v3_results.json",
        performance_eligible=False,
        builds=builds,
        tasks={
            "state": dict(
                task_id="task_20261023_135137_7079796761",
                output="runs/h131072_b24_s6_output_rectangle_v3_state",
                harness="uniform_state_harness_v1.json",
                expected_metadata_cases=16,
                expected_continuous_steps=4,
            ),
            "service": dict(
                task_id="task_20261023_135138_70859816005",
                output="runs/h131072_b24_s6_output_rectangle_v3_service",
                harness="priority_service_harness_v1.json",
            ),
        },
        adoption_requires="Runtime state/service, unchanged seven-case CPP/ABI and production integration checks",
    )
    with (here / "output_rectangle_v3_validation_tasks.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print("Prepared six missing descriptors; C04/C05 CPU gates retained")


if __name__ == "__main__":
    main()
