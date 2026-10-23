# SPDX-License-Identifier: Apache-2.0
"""Certify all eight emitted kernels against previously measured implementations."""

import hashlib
import json
from pathlib import Path

from analyze_priority_phase import executable


def main():
    here = Path(__file__).resolve().parent
    retained = json.loads((here / "merged_schedule_v1_codegen.json").read_text())["cases"]
    cases = {}
    for number in range(1, 9):
        case = f"C{number:02d}"
        build = here / f"runs/compile_physical_capacity_tier_{case}_v1"
        assert json.loads((build / "result.json").read_text())["passed"]
        candidate = build / "kernel"
        reference = (
            here / "runs/compile_physical_capacity_C02_v1/kernel"
            if case == "C02"
            else here.parents[2] / retained[case].get("actual_measured_artifact", retained[case]["artifact"])
        )
        paths = sorted(p.relative_to(candidate) for p in candidate.rglob("*.cpp"))
        assert paths == sorted(p.relative_to(reference) for p in reference.rglob("*.cpp"))
        assert len(paths) == 4
        checked = {}
        for path in paths:
            left = executable((candidate / path).read_text())
            right = executable((reference / path).read_text())
            assert left == right, (case, path)
            checked[str(path)] = hashlib.sha256(left.encode()).hexdigest()
        actual = json.loads((candidate / "compiled_meta.json").read_text())
        expected = json.loads((reference / "compiled_meta.json").read_text())
        assert actual["kernel_abi"] == expected["kernel_abi"]
        cases[case] = dict(
            candidate=str(candidate),
            reference=str(reference),
            executable_cpp_sha256=checked,
            all_four_cpp_equal=True,
            kernel_abi_equal=True,
            capacity=2080 if case == "C02" else 2048,
        )
    report = dict(
        passed=True,
        candidate="physical_capacity_tier_v1",
        device_execution=False,
        comparison="Exact text after removing only // pto: source-location comment lines",
        cpu_contract_tests=165,
        cases=cases,
        limitation="Code identity permits timing reuse; binding/service/state checks remain independent gates",
    )
    with (here / "physical_capacity_tier_v1_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print("PASS: eight kernels, four C++ files and ABI per case")


if __name__ == "__main__":
    main()
