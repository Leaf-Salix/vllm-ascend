# SPDX-License-Identifier: Apache-2.0
"""Boundary evidence must not masquerade as enabled math or fixed-cohort data."""

import copy
import json
from pathlib import Path

import pytest
from collect_query_guard_boundaries import measurement
from evaluate_nine_cases import case_measurement

HERE = Path(__file__).resolve().parent


def result():
    return json.loads((HERE / "runs/query_head_guard_t64/result.json").read_text())


def test_disabled_control_is_explicit_and_the_nine_case_gate_still_rejects_it():
    data = result()
    row = measurement(data)
    assert row["classification"] == "disabled control" and not row["enabled"]
    with pytest.raises(ValueError, match="fixed nine-case"):
        case_measurement(data, json.loads((HERE / "frozen_v122.json").read_text())["files"])


@pytest.mark.parametrize("corruption", ("flag", "shared", "cores", "order", "clock", "phase", "few_windows"))
def test_boundary_collection_rejects_wrong_activation_or_timing(corruption):
    data = copy.deepcopy(result())
    if corruption == "flag":
        data["policy"]["QUERY_CONTIGUOUS_HEADS"] = True
    elif corruption == "shared":
        data["l1"]["shared_workspace"] = True
    elif corruption == "cores":
        data["l1"]["windows"][0]["core_records"] = 71
    elif corruption == "order":
        data["l1"]["windows"][1]["side"] = data["l1"]["windows"][0]["side"]
    elif corruption == "clock":
        data["l1"]["windows"][0]["clock_freq_hz"] = 100000000
    elif corruption == "phase":
        data["phase_clock"] = {"enabled": True}
    else:
        data["l1"]["windows"] = data["l1"]["windows"][:8]
    with pytest.raises(AssertionError):
        measurement(data)
