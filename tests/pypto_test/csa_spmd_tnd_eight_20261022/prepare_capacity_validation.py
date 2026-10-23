# SPDX-License-Identifier: Apache-2.0
"""Freeze C02 state/service validation, preserving prior checks and SDK selection."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    for mode in ("state", "service"):
        parent = here / f"runs/m192_{mode}_harness_v1"
        target = here / f"runs/capacity_{mode}_harness_v1"
        changes = {}
        source = (parent / "run_pto.py").read_text()
        source = replace_once(
            source,
            'args.case not in ("h131072_b16_s6", "h131072_b24_s6", "h131072_b28_s6")',
            'args.case != "h262144_b16_s6"',
        )
        source = source.replace("C04/C05/C06", "C02")
        changes["run_pto.py"] = source
        if mode == "state":
            source = (parent / "state_replay.py").read_text()
            source = replace_once(
                source,
                "import torch\n",
                "import torch\nfrom capacity_state_observation import inspect_plan, invalidate_selected_pages\n",
            )
            source = replace_once(source, "            selected = select(\n", "            policy_args = dict(\n")
            source = replace_once(
                source,
                "            if selected != entry.policy:\n",
                '            if hasattr(entry.policy, "plan_table_columns"):\n'
                '                policy_args["compressed_table_columns"] = root["cmp_block_table"].shape[1]\n'
                "            selected = select(**policy_args)\n"
                "            if selected != entry.policy:\n",
            )
            source = replace_once(
                source,
                "        expected, next_caches = None, {}\n",
                "        hole, prepass = None, None\n"
                '        if label in ("compressed_negative_pages", "compressed_oob_pages"):\n'
                '            prepass, _, checks = run("reference", other, metadata, slots, previous["reference"])\n'
                '            assert checks["same_addresses"] and all(checks["guards"].values())\n'
                '            assert all(checks["generation"].values()) and all(checks["geometry"].values())\n'
                '            pages = entries["candidate"].buffers["main_cache"].shape[0] // 32\n'
                '            invalid = -1 if label == "compressed_negative_pages" else pages\n'
                '            hole = invalidate_selected_pages(metadata, prepass["topk"], histories, pages, invalid)\n'
                '            row["compressed_page_holes"] = hole\n'
                "        expected, next_caches = None, {}\n",
            )
            source = replace_once(
                source,
                '            checked["numerical"] = compare(values, expected)\n',
                '            checked["numerical"] = compare(values, expected)\n'
                "            if prepass is not None:\n"
                '                assert torch.equal(values["topk"], prepass["topk"])\n'
                '            if side == "candidate":\n'
                '                checked["physical_plan"] = inspect_plan(entries[side], root, hole)\n',
            )
            source = replace_once(
                source,
                "    for label, queries, history_values, owners, pattern in schedules:\n",
                '    schedules += [(label, [6] * requests, [history] * requests, None, "original")\n'
                '                  for label in ("compressed_negative_pages", "compressed_negative_recovery",\n'
                '                                "compressed_oob_pages", "compressed_oob_recovery")]\n'
                "    for label, queries, history_values, owners, pattern in schedules:\n",
            )
            changes["state_replay.py"] = source
            for filename in ("capacity_state_observation.py",):
                changes[filename] = (here / filename).read_text()
        else:
            source = (parent / "service_validation.py").read_text()
            source = replace_once(
                source,
                "    if not prepared.policy.scale_page_coalescing:\n"
                '        raise AssertionError("Canonical service did not select '
                'the continuous-scale leaf proof tier")\n',
                "    if prepared.policy.plan_table_columns != 2080 or not prepared.policy.plan_physical_rows:\n"
                '        raise AssertionError("Canonical service did not select the 2080-column physical plan")\n',
            )
            changes["service_validation.py"] = source
        for content in changes.values():
            ast.parse(content)
        shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename, content in changes.items():
            (target / filename).write_text(content)
        launcher = replace_once(
            (here / f"launch_m192_{mode}.sh").read_text(),
            f'harness="$here/runs/m192_{mode}_harness_v1"',
            f'harness="$here/runs/capacity_{mode}_harness_v1"',
        )
        with (here / f"launch_capacity_{mode}.sh").open("x") as stream:
            stream.write(launcher)
        report = dict(
            source=str(target),
            parent=f"m192_{mode}_harness_v1.json",
            performance_eligible=False,
            scope="C02 capacity tier same-address state and service integration",
            state_cases=20 if mode == "state" else None,
            continuous_steps=4 if mode == "state" else None,
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        )
        with (here / f"capacity_{mode}_harness_v1.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        print(target)


if __name__ == "__main__":
    main()
