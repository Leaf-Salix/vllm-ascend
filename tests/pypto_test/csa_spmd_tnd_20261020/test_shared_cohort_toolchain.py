# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete/shared SDK release mismatches in the eight-case gate."""

import json
from pathlib import Path

import pytest
from evaluate_s6_cohort import shared_toolchain_fingerprint


@pytest.mark.parametrize("fault", [None, "validating", "audit", "release", "revision", "build", "ambiguous"])
def test_shared_release_requires_complete_matching_evidence(tmp_path, fault):
    here = Path(__file__).resolve().parent
    data = json.loads((here.parent / "csa_shared_toolchain/manifest.json").read_text())
    result = {"toolchain_post_run_audit": {"passed": True, "release": data["release"]}}
    if fault == "validating":
        data["status"] = "validating"
    elif fault == "audit":
        result["toolchain_post_run_audit"]["passed"] = False
    elif fault == "release":
        result["toolchain_post_run_audit"]["release"] = "different-release"
    elif fault == "revision":
        data["pypto"].pop("commit")
    elif fault == "build":
        data["ptoas"].pop("build")
    elif fault == "ambiguous":
        (tmp_path / "native_toolchain_manifest.json").write_text("{}")
    (tmp_path / "shared_toolchain_manifest.json").write_text(json.dumps(data))
    if fault:
        with pytest.raises(ValueError):
            shared_toolchain_fingerprint(tmp_path, result)
    else:
        fingerprint = shared_toolchain_fingerprint(tmp_path, result)
        assert fingerprint["release"] == data["release"]
        assert len(fingerprint["manifest_sha256"]) == 64


def test_distinct_sdk_revisions_cannot_share_a_cohort_fingerprint(tmp_path):
    here = Path(__file__).resolve().parent
    data = json.loads((here.parent / "csa_shared_toolchain/manifest.json").read_text())
    result = {"toolchain_post_run_audit": {"passed": True, "release": data["release"]}}
    path = tmp_path / "shared_toolchain_manifest.json"
    path.write_text(json.dumps(data))
    original = shared_toolchain_fingerprint(tmp_path, result)
    data["pypto"]["commit"] = "0" * 40
    path.write_text(json.dumps(data))
    assert shared_toolchain_fingerprint(tmp_path, result) != original
