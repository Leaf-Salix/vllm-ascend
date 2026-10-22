# SPDX-License-Identifier: Apache-2.0
"""Ensure diagnostic instrumentation cannot change waits or enter the performance gate."""

import re
from pathlib import Path

import pytest
from evaluate_nine_cases import case_measurement
from phase_clock import CLOCK_HELPERS, instrument


def test_full_generated_module_roundtrip_keeps_original_operations():
    here = Path(__file__).resolve().parent
    path = here / "runs/compile_output_balance_off_h131072_b16/kernel/ptoas/dynamic_csa_full.cpp"
    if not path.exists():
        pytest.skip("Requires local generated current-SDK module")
    source = path.read_text()
    changed, audit = instrument(source)
    assert audit["eligible_for_performance"] is False
    assert audit["PTOAS__WAIT_GE_BACKOFF<150>"] > 0
    assert audit["pto::comm::TNOTIFY"] > 0
    restored = changed.replace(CLOCK_HELPERS + "\n", "", 1)
    restored = restored.replace("  uint64_t csa_wait_ticks = 0;\n  uint32_t csa_wait_count = 0;\n", "")
    for new, old in (("CSA_PROFILE_WAIT", "PTOAS__WAIT_GE_BACKOFF<150>"), ("CSA_PROFILE_NOTIFY", "pto::comm::TNOTIFY")):
        restored = re.sub(
            re.escape(new) + r"\(([^\n]+), csa_wait_ticks, csa_wait_count\);",
            lambda m, old=old: old + "(" + m[1] + ");",
            restored,
        )
    assert restored == source
    with pytest.raises(ValueError, match="double instrumentation"):
        instrument(changed)
    with pytest.raises(ValueError, match="native150"):
        instrument(source.replace("PTOAS__WAIT_GE_BACKOFF<150>", "PTOAS__WAIT_GE_BACKOFF<100>"))


@pytest.mark.parametrize("marker", ["result", "reference"])
def test_instrumented_result_cannot_pass_performance_gate(marker):
    result = (
        {"phase_clock": {"eligible_for_performance": False}}
        if marker == "result"
        else {"comparison_reference": {"diagnostic_phase_clock": True}}
    )
    with pytest.raises(ValueError, match="Instrumented phase clocks"):
        case_measurement(result, {})
