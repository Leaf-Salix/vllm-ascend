# SPDX-License-Identifier: Apache-2.0
"""Validate actual SDK/source identities before comparing full-L1 samples."""

import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path


def require_same_sdk(actual, reference):
    if actual.get("status") != "validated" or reference.get("status") != "validated":
        raise ValueError("Both SDK snapshots must be validated")
    keys = ["release", "pypto", "ptoas", "simpler"]
    for key in keys:
        left = actual[key] if key == "release" else actual[key]["commit"]
        right = reference[key] if key == "release" else reference[key]["commit"]
        if left != right:
            raise ValueError(f"Cross-SDK comparison rejected: {key}: {left} != {right}")


def capture_toolchain(expected):
    """Read the shared selection and three Git HEADs once; never hash the SDK tree."""
    import pypto

    manifest = Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).resolve()
    current = json.loads(manifest.read_text())
    require_same_sdk(current, expected)
    if os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"] != current["release"]:
        raise ValueError("SDK selection changed between bootstrap and measurement")
    workspace = next(p for p in manifest.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    paths = {}
    for name in ("pypto", "ptoas", "simpler"):
        selected = (workspace / current[name]["source"]).resolve()
        actual = Path(os.environ[f"CSA_SHARED_{name.upper()}_ROOT"]).resolve()
        expected_root = (workspace / current[name].get("build", current[name]["source"])).resolve()
        if actual != expected_root:
            raise ValueError(f"Actual {name} path differs from manifest")
        head = subprocess.check_output(["git", "-C", str(selected), "rev-parse", "HEAD"], text=True).strip()
        if head != current[name]["commit"]:
            raise ValueError(f"Actual {name} HEAD differs from manifest")
        paths[name] = str(actual)
    imported = Path(pypto.__file__).resolve()
    if not imported.is_relative_to(paths["pypto"]):
        raise ValueError(f"Imported PyPTO is outside the selected SDK: {imported}")
    return {**current, "manifest": str(manifest), "actual_paths": paths, "pypto_import": str(imported)}


def package_hashes(source):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))}


def validate_sources(reference, candidate, frozen, calibrating):
    hashes = {"reference": package_hashes(reference), "candidate": package_hashes(candidate)}
    if hashes["reference"] != frozen["files"]:
        raise ValueError("Correctness reference differs from the frozen baseline implementation")
    if calibrating and hashes["candidate"] != frozen["files"]:
        raise ValueError("SDK calibration must measure the unchanged frozen baseline implementation")
    return hashes


def validated_reference(report, case, sdk, frozen):
    require_same_sdk(sdk, report["shared_toolchain"])
    if not report.get("passed") or not report.get("calibration") or report["case"] != case:
        raise ValueError("Expected a passing calibration of the exact case")
    if report["comparison_reference"] != frozen["reference_kind"]:
        raise ValueError("Baseline algorithm identity changed")
    if report["source_hashes"]["candidate"] != frozen["files"]:
        raise ValueError("Calibration source is not the frozen baseline")
    windows = report["windows"]
    if len(windows) < 10 or any(
        w["side"] != "candidate" or w["cores"] != 72 or w["records"] != 72 or w["task_groups"] != 1
        for w in windows
    ):
        raise ValueError("Reference needs at least ten complete single-MIX L1 windows")
    samples = [w["incore_span_us"] for w in windows]
    if not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError("Invalid reference timestamps")
    stats = report["statistics_us"]["candidate"]
    if samples != stats["samples"] or statistics.median(samples) != stats["median"]:
        raise ValueError("Reference statistics do not match all raw windows")
    return stats["median"]
