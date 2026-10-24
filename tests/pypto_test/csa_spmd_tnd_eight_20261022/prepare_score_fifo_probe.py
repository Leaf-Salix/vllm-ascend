# SPDX-License-Identifier: Apache-2.0
"""Freeze a diagnostic harness; retain production math, ABI and shared SDK."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/score_fifo_harness_v1"
    shutil.copytree(here / "runs/phase_sdk_harness_v1", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(here / "score_fifo_probe.py", target / "score_fifo_probe.py")
    hooks = (target / "phase_probe_hooks.py").read_text()
    hooks = replace_once(
        hooks, "from phase_clock import instrument", "from score_fifo_probe import export_fifo, instrument"
    )
    hooks = replace_once(
        hooks,
        "    assert not signal[:, :, 1:16].count_nonzero() and not signal[:, :, 20:].count_nonzero()\n"
        "    return export_clock(signal, directory, phases)",
        "    assert not signal[:, :, 1:16].count_nonzero()\n"
        "    assert not signal[:, :, 20:32].count_nonzero() and not signal[:, :, 38:].count_nonzero()\n"
        "    result = export_clock(signal, directory, phases)\n"
        "    result['score_fifo'] = export_fifo(\n"
        "        signal, directory, phases, plan=entry.buffers['group_plan'].cpu().tolist(),\n"
        "        offsets=entry.buffers['group_offsets'].cpu().tolist(),\n"
        "        requests=entry.buffers['idx_lengths'].numel())\n"
        "    return result",
    )
    (target / "phase_probe_hooks.py").write_text(hooks)
    launcher = replace_once(
        (here / "launch_priority_phase.sh").read_text(),
        'harness="$here/runs/phase_sdk_harness_v1"',
        'harness="$here/runs/score_fifo_harness_v1"',
    )
    (here / "launch_score_fifo_probe.sh").write_text(launcher)
    sys.path.insert(0, str(target))
    from score_fifo_probe import instrument

    audits = {}
    for case in ("C04", "C05"):
        cpp = here / f"runs/compile_output_rectangle_production_{case}_v1/kernel/ptoas/dynamic_csa_full.cpp"
        raw = cpp.read_text()
        revised, audit = instrument(raw)
        (target / f"{case}_instrumented.cpp").write_text(revised)
        audits[case] = dict(raw=str(cpp), original_sha256=hashlib.sha256(raw.encode()).hexdigest(), audit=audit)
    report = dict(
        performance_eligible=False,
        source="output_rectangle_production_v1",
        cases=audits,
        purpose="Resolve current Score producer/consumer pressure before redesigning dense key supply",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        shared_toolchain=json.loads((target / "expected_sdk.json").read_text()),
    )
    with (here / "score_fifo_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({case: value["audit"] for case, value in audits.items()}, indent=2))


if __name__ == "__main__":
    main()
