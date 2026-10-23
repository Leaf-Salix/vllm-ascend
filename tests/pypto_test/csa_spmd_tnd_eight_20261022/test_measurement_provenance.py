# SPDX-License-Identifier: Apache-2.0
"""Prevent cross-release gains and substitution of a different baseline implementation."""

import ast
import copy
import hashlib
from pathlib import Path

import pytest
from measurement_provenance import require_same_sdk, validate_sources, validated_reference
from prepare_sdk_measurement import transform


def fixture():
    sdk = dict(
        release="validated-release", status="validated", **{n: {"commit": n} for n in ("pypto", "ptoas", "simpler")}
    )
    frozen = dict(reference_kind="unchanged-baseline", files={"entry.py": "original"})
    case = dict(id="C04", query_lengths=[6] * 16)
    windows = [
        dict(side="candidate", cores=72, records=72, task_groups=1, incore_span_us=float(i + 1)) for i in range(10)
    ]
    report = dict(
        passed=True,
        calibration=True,
        case=case,
        shared_toolchain=copy.deepcopy(sdk),
        comparison_reference=frozen["reference_kind"],
        source_hashes={"candidate": frozen["files"].copy()},
        windows=windows,
        statistics_us={"candidate": {"samples": [w["incore_span_us"] for w in windows], "median": 5.5}},
    )
    return sdk, frozen, case, report


@pytest.mark.parametrize("field", ["release", "status", "pypto", "ptoas", "simpler"])
def test_reject_different_or_unvalidated_sdk(field):
    sdk, _, _, report = fixture()
    other = report["shared_toolchain"]
    if field in ("release", "status"):
        other[field] = "different"
    else:
        other[field]["commit"] = "different"
    with pytest.raises(ValueError):
        require_same_sdk(sdk, other)


def test_accept_exact_source_case_sdk_and_all_ten_windows():
    sdk, frozen, case, report = fixture()
    assert validated_reference(report, case, sdk, frozen) == 5.5


@pytest.mark.parametrize("mutation", ["source", "case", "window_count", "cores", "median", "samples", "passed"])
def test_reject_unqualified_reference(mutation):
    sdk, frozen, case, report = fixture()
    if mutation == "source":
        report["source_hashes"]["candidate"]["entry.py"] = "older-implementation"
    elif mutation == "case":
        report["case"] = dict(id="C04", query_lengths=[1, 6] * 8)
    elif mutation == "window_count":
        report["windows"].pop()
    elif mutation == "cores":
        report["windows"][0]["cores"] = 71
    elif mutation == "median":
        report["statistics_us"]["candidate"]["median"] = 1.0
    elif mutation == "samples":
        report["statistics_us"]["candidate"]["samples"][0] = 0.5
    else:
        report["passed"] = False
    with pytest.raises(ValueError):
        validated_reference(report, case, sdk, frozen)


def test_calibration_cannot_replace_frozen_algorithm(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for path in (reference, candidate):
        path.mkdir()
        (path / "entry.py").write_text("original")
    frozen = dict(files={"entry.py": hashlib.sha256(b"original").hexdigest()})
    assert validate_sources(reference, candidate, frozen, True)["candidate"] == frozen["files"]
    (candidate / "entry.py").write_text("slower alternative")
    with pytest.raises(ValueError, match="unchanged frozen baseline"):
        validate_sources(reference, candidate, frozen, True)
    assert validate_sources(reference, candidate, frozen, False)["candidate"] != frozen["files"]


def test_sdk_patch_preserves_actual_measurement_and_correctness_loops():
    path = Path(__file__).with_name("runs") / "measurement_harness/run_pto.py"
    if not path.is_file():
        pytest.skip("Frozen predecessor harness unavailable")
    before = ast.parse(path.read_text())
    after = ast.parse(transform(path.read_text()))
    for name in ("export_l1", "snapshot", "compare"):
        nodes = [
            next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            for tree in (before, after)
        ]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    for span in ("range(3)", "range(10)"):
        loops = [
            next(n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.iter) == span)
            for tree in (before, after)
        ]
        assert ast.dump(loops[0]) == ast.dump(loops[1])
