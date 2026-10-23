# SPDX-License-Identifier: Apache-2.0
"""Freeze a 2080-column physical-plan prototype for the 2050-column C02 case."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    name = "attention_plan_stages.py"
    source[name] = replace_once(source[name], "PHYSICAL_PLAN_COLUMNS = 2048", "PHYSICAL_PLAN_COLUMNS = 2080")
    for content in source.values():
        ast.parse(content)
    target = here / "runs/physical_capacity_2080_v1"
    target.mkdir(exist_ok=False)
    for filename, content in source.items():
        (target / filename).write_text(content)
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        tested_scope="C02 256K/B16/S6 only; not a production-wide capacity change",
        mechanism="Allow the existing physical plan and MGATHER path for the actual 2050-column table",
        changed={name: "PHYSICAL_PLAN_COLUMNS 2048 -> 2080; shared producer/consumer predicate unchanged"},
        unchanged="Math, TopK ordering, causal/page guards, exact integer address arithmetic, ABI and event protocol",
        gate="Actual C02 allocation/codegen, boundary address tests, strict full-L1 comparison and path witness",
        integration_requirement="If promising, select the extended resource tier without altering the old 2048 tier",
        files={filename: hashlib.sha256(content.encode()).hexdigest() for filename, content in source.items()},
    )
    with (here / "physical_capacity_2080_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
