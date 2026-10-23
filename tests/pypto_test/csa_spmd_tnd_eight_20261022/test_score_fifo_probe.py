# SPDX-License-Identifier: Apache-2.0
"""Guard diagnostic transparency and reject incorrect protocol observations."""

import json
import re
import sys
from pathlib import Path

import pytest
from score_fifo_probe import DECLARATIONS, expected_steps, export_fifo, instrument

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runs/phase_sdk_harness_v1"))


@pytest.mark.parametrize("case", ["C04", "C05"])
def test_removing_probe_recovers_exact_native_cpp(case):
    raw = (HERE / f"runs/compile_output_rectangle_production_{case}_v1/kernel/ptoas/dynamic_csa_full.cpp").read_text()
    modified, audit = instrument(raw)
    marker = "AICORE void cube_schedule("
    prefix, body = modified.split(marker, 1)
    raw_prefix = raw.split(marker, 1)[0]
    assert prefix.startswith(raw_prefix)
    # Helpers only precede the unchanged generated functions. Reverse call-site
    # changes and require byte equality for the entire generated body.
    body = re.sub(
        r"CSA_SCORE_FIFO_WAIT\((v\d+), csa_ready_ticks, csa_ack_ticks, csa_ready_count, csa_ack_count\);",
        r"wait_flag_dev(\1);",
        body,
    )
    body = re.sub(
        r"CSA_PROFILE_NOTIFY\(([^\n]+), csa_wait_ticks, csa_wait_count, "
        r"csa_ready_ticks, csa_ack_ticks, csa_ready_count, csa_ack_count, csa_score_flushed\);",
        r"pto::comm::TNOTIFY(\1);",
        body,
    )
    body = re.sub(
        r"CSA_PROFILE_WAIT\(([^\n]+), csa_wait_ticks, csa_wait_count\);", r"PTOAS__WAIT_GE_BACKOFF<150>(\1);", body
    )
    body = body.replace("  uint64_t csa_wait_ticks = 0;\n  uint32_t csa_wait_count = 0;\n" + DECLARATIONS, "")
    assert raw_prefix + marker + body == raw
    assert audit["fifo_wait_source_sites"] == 10 and not audit["eligible_for_performance"]
    with pytest.raises(ValueError, match="Already instrumented"):
        instrument(modified)


@pytest.mark.parametrize("groups,leaves,tiles,extra,steps", [(16, 6, 5, 3, 22), (24, 5, 8, 0, 33)])
def test_counts_include_final_partial_step_and_drain(groups, leaves, tiles, extra, steps):
    base = ((groups + 16) // 16) * 16
    offsets = [0] * (base + 4 * groups)
    for group in range(groups):
        offsets[base + 4 * group + 3] = 32769
    plan = [groups, 32769, leaves, tiles, extra, 6, 1, 1]
    assert expected_steps(plan, offsets, groups) == [steps] * 24
    offsets[base + 3] = 0
    counts = expected_steps(plan, offsets, groups)
    assert sum(counts) == (groups - 1) * 33


class Signal:
    def __init__(self, slots):
        self.slots = slots

    def tolist(self):
        return self.slots


def fixture():
    plan = [24, 32769, 5, 8, 0, 6, 1, 1]
    offsets = [0] * (32 + 4 * 24)
    for group in range(24):
        offsets[32 + 4 * group + 3] = 32769
    signal = Signal([[[0] * 128 for _ in range(72)] for _ in range(2)])
    for core in range(72):
        phase, word = (0, 35) if core < 24 else (1, 32)
        signal.slots[phase][core][word : word + 3] = [50, 1, 33]
    return signal, plan, offsets


def test_full_72_role_export_and_64bit_ticks(tmp_path):
    signal, plan, offsets = fixture()
    result = export_fifo(
        signal, tmp_path, {0: "SCORE_CUBE", 1: "SCORE_VECTOR"}, plan=plan, offsets=offsets, requests=24
    )
    saved = json.loads(Path(result["raw"]).read_text())
    assert saved["passed"] and len(saved["rows"]) == 72 and not saved["performance_eligible"]
    assert saved["metrics"][0]["median_us"] == ((1 << 32) + 50) / 50


@pytest.mark.parametrize("bad", ["phase", "count", "counterpart"])
def test_reject_corrupt_protocol_dump(tmp_path, bad):
    signal, plan, offsets = fixture()
    if bad == "phase":
        signal.slots[1][0][32] = 1
    elif bad == "count":
        signal.slots[0][0][37] -= 1
    else:
        signal.slots[0][0][34] = 1
    with pytest.raises(AssertionError):
        export_fifo(signal, tmp_path, {0: "SCORE_CUBE", 1: "SCORE_VECTOR"}, plan=plan, offsets=offsets, requests=24)
