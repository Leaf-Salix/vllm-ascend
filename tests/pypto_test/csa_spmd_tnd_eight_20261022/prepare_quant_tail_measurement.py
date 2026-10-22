# SPDX-License-Identifier: Apache-2.0
"""Check unread quant padding remains poisoned, outside all timed windows."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/quant_tail_measurement_v1"
    shutil.copytree(here / "runs/sdk_measurement_harness_v1", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "run_pto.py"
    source = path.read_text()
    marker = "                return state, checks\n"
    source = replace_once(source, marker,
        '                if side == "candidate":\n'
        '                    entry = entries[side]\n'
        '                    assert entry.policy.ob_rows > 96\n'
        '                    actual_t = entry.buffers["x"].shape[0]\n'
        '                    unused = entry.buffers["o_quant"][actual_t:].cpu()\n'
        '                    assert unused.numel() > 0\n'
        '                    checks["scratch/o_quant_unread_tail_poison"] = bool((unused == 37).all())\n'
        + marker)
    ast.parse(source)
    path.write_text(source)
    launcher = replace_once((here / "launch_sdk_measure.sh").read_text(),
                            'harness="$here/runs/sdk_measurement_harness_v1"',
                            'harness="$here/runs/quant_tail_measurement_v1"')
    (here / "launch_quant_tail_measure.sh").write_text(launcher)
    report = dict(
        parent="sdk_measurement_harness_v1", source=str(target),
        added_check="candidate o_quant[T:] remains initial INT8 37 after eager, graph and all timing windows",
        mathematical_oracle="Existing frozen full output/TopK/score/cache/state bit checks unchanged",
        scope="Nonempty unused padding in NZ large-M policy only; no change to the public eight-case inputs",
        sampling_unchanged=True, measurement_boundary_unchanged=True,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "quant_tail_measurement_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
