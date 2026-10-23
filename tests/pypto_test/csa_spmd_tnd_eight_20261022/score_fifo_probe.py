# SPDX-License-Identifier: Apache-2.0
"""Diagnostic Scalar intervals around the retained Score FIFO's original waits.

This hook is applied in a private diagnostic process, after the native150 phase
hook. It neither changes SDK files nor supplies performance-gain samples.
"""

import json
import re
import statistics
from pathlib import Path

COUNTER_ARGS = "csa_ready_ticks, csa_ack_ticks, csa_ready_count, csa_ack_count"
DECLARATIONS = """
  uint64_t csa_ready_ticks = 0, csa_ack_ticks = 0;
  uint32_t csa_ready_count = 0, csa_ack_count = 0;
  bool csa_score_flushed = false;
"""
HELPER = r"""
static AICORE inline void CSA_SCORE_FIFO_WAIT(int32_t event,
    uint64_t& ready_ticks, uint64_t& ack_ticks,
    uint32_t& ready_count, uint32_t& ack_count) {
  if (event != 0 && event != 1) {
    wait_flag_dev(event);
    return;
  }
  const uint64_t before = get_sys_cnt();
  wait_flag_dev(event);
  const uint64_t delta = get_sys_cnt() - before;
  if (event == 0) { ready_ticks += delta; ready_count += 1; }
  else { ack_ticks += delta; ack_count += 1; }
}
"""
STORE = r"""
  // Third cache line of the role's own phase slot; original signal is untouched.
  if (!score_flushed && (ready_count + ack_count) > 0) {
    ptr[32] = static_cast<uint32_t>(ready_ticks);
    ptr[33] = static_cast<uint32_t>(ready_ticks >> 32);
    ptr[34] = ready_count;
    ptr[35] = static_cast<uint32_t>(ack_ticks);
    ptr[36] = static_cast<uint32_t>(ack_ticks >> 32);
    ptr[37] = ack_count;
    dcci((__gm__ void*)(ptr + 32), cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    score_flushed = true;
  }
"""


def instrument(source):
    from phase_clock import CLOCK_HELPERS
    from phase_clock import instrument as phase_instrument

    if "CSA_SCORE_FIFO_WAIT" in source:
        raise ValueError("Already instrumented")
    revised, audit = phase_instrument(source)
    signature = "pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count)"
    assert CLOCK_HELPERS.count(signature) == 1
    helper = CLOCK_HELPERS.replace(
        signature,
        "pto::comm::NotifyOp op, uint64_t total_ticks, uint32_t count,\n"
        "    uint64_t ready_ticks, uint64_t ack_ticks,\n"
        "    uint32_t ready_count, uint32_t ack_count, bool& score_flushed)",
    )
    assert helper.count("  dsb(DSB_DDR);\n}") == 1
    helper = helper.replace("  dsb(DSB_DDR);\n}", "  dsb(DSB_DDR);\n" + STORE + "}")
    assert revised.count(CLOCK_HELPERS) == 1
    revised = revised.replace(CLOCK_HELPERS, helper)
    declaration = "  uint32_t csa_wait_count = 0;\n"
    assert revised.count(declaration) == 2
    revised = revised.replace(declaration, declaration + DECLARATIONS)
    pattern = re.compile(r"^(\s*)wait_flag_dev\((v\d+)\);$", re.M)
    revised, count = pattern.subn(lambda m: f"{m[1]}CSA_SCORE_FIFO_WAIT({m[2]}, {COUNTER_ARGS});", revised)
    if count != 10:
        raise ValueError(f"Expected 10 native wait sites in priority retained code, found {count}")
    pattern = re.compile(r"^(\s*)CSA_PROFILE_NOTIFY\(([^\n]+)\);$", re.M)
    revised, notify_count = pattern.subn(
        lambda m: f"{m[1]}CSA_PROFILE_NOTIFY({m[2]}, {COUNTER_ARGS}, csa_score_flushed);", revised
    )
    assert notify_count == audit["pto::comm::TNOTIFY"]
    revised = revised.replace("AICORE void cube_schedule(", HELPER + "\nAICORE void cube_schedule(")
    audit.update(
        score_fifo_event_ids=[0, 1],
        fifo_wait_source_sites=count,
        fifo_words=[32, 38],
        fifo_dump_once_per_role=True,
        scope="Scalar observation around original waits; asynchronous issue/dependency overlap is not separated",
        eligible_for_performance=False,
    )
    return revised, audit


def expected_steps(plan, offsets, requests):
    """Reconstruct counts from this invocation's device descriptors, not B*S."""
    groups, _, leaves, leaf_tiles, extra, group_size, balanced, _ = plan
    if group_size != 6 or not balanced or groups not in (16, 24):
        raise ValueError("Probe exporter is limited to the two priority balanced-G6 cases")
    base = ((requests + 1 + 15) // 16) * 16
    counts = [0] * 24
    for item in range(groups * leaves):
        group, leaf = (item % groups, item // groups) if groups < 24 else divmod(item, leaves)
        visible = offsets[base + group * 4 + 3]
        begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
        capacity = (leaf_tiles + int(leaf < extra)) * 1024
        valid = max(0, min(capacity, visible - begin))
        counts[item % 24] += (valid + 1023) // 1024
    return counts


def export_fifo(signal, destination, phases, *, plan, offsets, requests):
    expected = expected_steps(plan, offsets, requests)
    rows = []
    for phase, cores in enumerate(signal.tolist()):
        name = phases.get(phase, "")
        for core, values in enumerate(cores):
            words = [int(value) & 0xFFFFFFFF for value in values[32:38]]
            correct_phase = name == ("SCORE_CUBE" if core < 24 else "SCORE_VECTOR")
            if not correct_phase:
                if any(words):
                    raise AssertionError(f"FIFO counters outside Score release: phase={phase}, core={core}")
                continue
            group = core if core < 24 else (core - 24) // 2
            counts = [words[2], words[5]]
            wanted = [0, expected[group]] if core < 24 else [expected[group], 0]
            if counts != wanted:
                raise AssertionError(f"FIFO protocol count mismatch: core={core}, actual={counts}, expected={wanted}")
            ticks = [(words[1] << 32) | words[0], (words[4] << 32) | words[3]]
            rows.append(
                dict(
                    core=core,
                    role="AIC" if core < 24 else "AIV",
                    group=group,
                    counts=counts,
                    ticks=ticks,
                    observed_us=[value / 50 for value in ticks],
                )
            )
    assert sorted(row["core"] for row in rows) == list(range(72))
    metrics = []
    for role, event in (("AIC", 1), ("AIV", 0)):
        selected = [row for row in rows if row["role"] == role]
        values = [row["observed_us"][event] for row in selected]
        metrics.append(
            dict(
                role=role,
                event="ACK" if event else "READY",
                roles=len(selected),
                min_us=min(values),
                median_us=statistics.median(values),
                max_us=max(values),
            )
        )
    result = dict(
        passed=True,
        performance_eligible=False,
        clock_freq_hz=50000000,
        rows=rows,
        metrics=metrics,
        plan=plan,
        expected_steps=expected,
        descriptor_prefix=offsets[: ((requests + 16) // 16) * 16 + 4 * plan[0]],
        scope="Instrumented Scalar intervals, not exclusive pipe stall time or recoverable full-L1 time",
    )
    path = Path(destination) / "score_fifo_wait.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return dict(raw=str(path), passed=True, roles=72, metrics=metrics)
