# SPDX-License-Identifier: Apache-2.0
"""Reuse established same-address validation for the actual C04/C05 capacities."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/priority_state_harness_v1"
    shutil.copytree(here / "runs/group_state_harness", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("measurement_provenance.py", "expected_sdk.json"):
        shutil.copy2(here / "runs/sdk_measurement_harness_v1" / name, target / name)
    source = (target / "run_pto.py").read_text()
    source = replace_once(
        source,
        "import types\n",
        "import types\n\nfrom measurement_provenance import capture_toolchain, validate_sources\n",
    )
    source = replace_once(
        source,
        '    frozen_baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())\n',
        "    import pypto\n    pypto.configure_cache(pypto.CacheConfig(enabled=False))\n"
        '    frozen_baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())\n',
    )
    source = replace_once(
        source,
        'args.case != "h131072_b16_mixed"',
        'args.case not in ("h131072_b16_s6", "h131072_b24_s6")',
    )
    source = source.replace("fixed C08 SPMD pair", "priority C04/C05 SPMD pair")
    source = replace_once(
        source,
        '        "performance_eligible": False,\n',
        '        "performance_eligible": False,\n'
        '        "shared_toolchain": capture_toolchain(\n'
        '            json.loads(Path(__file__).with_name("expected_sdk.json").read_text())),\n'
        '        "source_hashes": validate_sources(\n'
        '            args.frozen / "pypto/deepseek_v4_flash_csa_spmd_tnd", args.candidate, frozen_baseline, False),\n',
    )
    (target / "run_pto.py").write_text(source)
    original_validator = here.parent / "csa_mpmd_tnd_20261021/state_replay.py"
    source = original_validator.read_text().replace('"spmd"', '"reference"').replace('"mpmd"', '"candidate"')
    source = replace_once(
        source,
        '    schedules = [(*schedule, "original") for schedule in schedules]\n',
        "    schedules += [\n"
        '        ("priority_all_s6", [6] * requests, [history] * requests, None),\n'
        '        ("priority_reverse_owners", [6] * requests, [history] * requests, list(reversed(range(requests)))),\n'
        "    ]\n"
        '    schedules = [(*schedule, "original") for schedule in schedules]\n',
    )
    (target / "state_replay.py").write_text(source)
    assert "score_order" not in source
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    assert "pypto.torch.begin_dfx()" not in (target / "run_pto.py").read_text()
    launcher = replace_once(
        (here / "launch_group_state.sh").read_text(),
        'harness="$here/runs/group_state_harness"',
        'harness="$here/runs/priority_state_harness_v1"',
    )
    (here / "launch_priority_state.sh").write_text(launcher)
    report = dict(
        source_validator=str(original_validator),
        purpose="Same-address metadata at priority capacities and four continuous steps; no performance samples",
        performance_eligible=False,
        metadata_cases=12,
        continuous_steps=4,
        additions=[
            "all requests S6 at full priority capacity",
            "reverse physical page owners at full priority capacity",
        ],
        shared_toolchain=json.loads((target / "expected_sdk.json").read_text()),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "priority_state_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
