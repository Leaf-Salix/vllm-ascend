# SPDX-License-Identifier: Apache-2.0
"""Reuse the measured fixture and L1 exporter; sample only the new SPMD candidate.

The frozen reference is executed for strict numerical checks, graph replay and
interleaving. Its historical ten-window medians remain the performance denominator.
"""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def transform(source):
    """Keep fixture creation, restore points and device measurement boundaries intact."""
    source = replace_once(
        source,
        '    if side == "spmd" and (len(rows) != 72 or len(task_ids) != 1):\n'
        '        raise ValueError("冻结 SPMD 基线不再是完整 MIX SPMD")\n'
        '    if side.startswith("mpmd") and len(task_ids) <= 1:\n'
        '        raise ValueError("MPMD 必须包含多个独立调度 task")\n',
        "    if len(rows) != 72 or len({r[0] for r in rows}) != 72 or len(task_ids) != 1:\n"
        '        raise ValueError("Expected one complete 72-role MIX task")\n',
    )
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        '    if args.family != "tnd" or args.control_mpmd or args.baseline_only '
        "or args.state_validation or args.service_validation:\n"
        '        raise ValueError("This frozen harness measures an independent TND-SPMD candidate only")\n'
        '    frozen_baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())\n',
    )
    source = replace_once(
        source,
        '        "scope": "full CSA L1",\n',
        '        "scope": "TND-SPMD full CSA L1; frozen reference medians reused",\n'
        '        "comparison_reference": frozen_baseline["reference_kind"],\n'
        '        "historical_reference": frozen_baseline["cases"][case["id"]]["cell"],\n',
    )
    source = replace_once(
        source,
        "                return result\n",
        "                for value in result.values():\n"
        '                    if not value["bit_equal"]:\n'
        '                        value["status"] = "FAIL"\n'
        "                return result\n",
    )
    source = replace_once(
        source,
        '                save("paired_l1")\n',
        '                save("candidate_l1_with_unmeasured_reference_replays")\n',
    )
    source = replace_once(
        source,
        "                        pypto.torch.begin_dfx()\n",
        '                        if side == "spmd":\n'
        "                            graphs[side].replay()\n"
        "                            stream.synchronize()\n"
        "                            continue\n"
        "                        pypto.torch.begin_dfx()\n",
    )
    begin = source.index('            if args.family == "tnd":\n                from outer_profile')
    end = source.index('            report["windows"] = records', begin)
    source = source[:begin] + source[end:]
    source = replace_once(source, "            for side in entries:\n", '            for side in ("mpmd",):\n')
    begin = source.index('            if "mpmd" in entries:\n')
    end = source.index('            report["passed"] = True\n', begin)
    source = (
        source[:begin]
        + (
            '            report["ratio_to_frozen"] = (report["statistics_us"]["mpmd"]["median"]\n'
            '                / report["historical_reference"]["median_us"])\n'
        )
        + source[end:]
    )
    source = source.replace('"spmd"', '"reference"').replace('"mpmd"', '"candidate"')
    source = source.replace("module.decode_csa_tnd", "module.decode_csa_spmd")
    ast.parse(source)
    return source


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    source_report = json.loads(Path(baseline["cases"]["C04"]["source_report"]).read_text())
    previous = Path(source_report["sources"]["spmd"]).parents[1]
    target = here / "runs/measurement_harness"
    shutil.copytree(previous, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    original = (previous / "run_pto.py").read_text()
    revised = transform(original)
    (target / "run_pto.py").write_text(revised)
    shutil.copyfile(here / "baseline.json", target / "baseline.json")
    for name, expected in baseline["files"].items():
        assert (
            hashlib.sha256((target / "pypto/deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
            == expected
        )
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(True), revised.splitlines(True), fromfile="a/run_pto.py", tofile="b/run_pto.py", n=0
        )
    )
    with (here / "measurement_harness.patch").open("x") as stream:
        stream.write(patch)
    report = dict(
        source=str(previous),
        target=str(target),
        baseline="baseline.json",
        source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_sha256=hashlib.sha256(revised.encode()).hexdigest(),
        samples=dict(candidate=10, reference=0, baseline_replays_for_correctness_and_interleaving=True),
        reference_medians_reused=True,
        exact_candidate_regression=True,
        note="No device data are collected while preparing this harness",
    )
    with (here / "measurement_harness.json").open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
