# SPDX-License-Identifier: Apache-2.0
"""在227的CPU测试，防止漏替换、回退或计时不足被判为验收通过。"""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("threeway_postflight_test", Path(__file__).with_name("postflight.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(arm="ours"):
    pto = arm != "native"
    layers = [
        {
            "layer": index,
            "name": f"layer{index}",
            "kind": "csa" if index < 21 else "hca",
            "implementation": "pto" if pto and index < 21 else "native",
            "runtime_class": "ops.deepseek_v4_flash_csa_spmd_tnd.service" if pto and index < 21 else None,
        }
        for index in range(41)
    ]
    return {
        "attention_implementations": [layers],
        "observation": [
            {
                "tnd_bindings": [
                    {
                        "layer": index,
                        "bindings": [{"policy": {}}],
                        "decode_graph_checks": {"allowed": 30, "rejected": 0},
                        "decode_submissions": 0,
                    }
                    for index in range(21)
                    if pto
                ],
                "capture_time_selection": {f"layer{i}": {"pto_tokens6": 1, "pto_tokens240": 1} for i in range(21)},
                "tnd_execution_routes": [
                    {
                        "actual_requests": 40,
                        "padded_tokens": 240,
                        "requested_mode": "FULL",
                        "all_csa_graph_replayed": True,
                        "python_resubmitted": False,
                    }
                ]
                if pto
                else [],
                "threeway_timing": {"complete": True},
            }
        ],
        "requests": [{"key": str(i), "output_token_ids": [1] * 192, "finished": True} for i in range(40)],
    }


@pytest.mark.parametrize("arm", ["native", "ours", "nalinaly"])
def test_complete_coverage_accepted(arm):
    assert not any(MODULE.check_rank(record(arm), arm).values())


def test_legal_first_non_full_submission_is_allowed():
    data = record()
    for row in data["observation"][0]["tnd_bindings"]:
        row["decode_submissions"] = 1
        row["decode_graph_checks"]["rejected"] = 1
    data["observation"][0]["tnd_execution_routes"].append(
        {
            "actual_requests": 0,
            "padded_tokens": 240,
            "requested_mode": "FULL",
            "all_csa_graph_replayed": False,
            "python_resubmitted": False,
        }
    )
    assert not any(MODULE.check_rank(data, "ours").values())


@pytest.mark.parametrize("failure", ["missing_runtime", "fallback", "missing_capture", "resubmit", "short_timing"])
def test_partial_coverage_rejected(failure):
    data = record()
    observed = data["observation"][0]
    if failure == "missing_runtime":
        observed["tnd_bindings"].pop()
    elif failure == "fallback":
        data["attention_implementations"][0][0]["implementation"] = "native"
    elif failure == "missing_capture":
        del observed["capture_time_selection"]["layer0"]["pto_tokens240"]
    elif failure == "resubmit":
        observed["tnd_execution_routes"][0]["python_resubmitted"] = True
    else:
        observed["threeway_timing"]["complete"] = False
    errors = MODULE.check_rank(data, "ours")
    assert errors["timing" if failure == "short_timing" else "coverage"]
