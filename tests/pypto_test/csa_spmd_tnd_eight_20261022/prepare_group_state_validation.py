# SPDX-License-Identifier: Apache-2.0
"""Adapt the existing same-address/continuous-state validator for this SPMD pair."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once

OBSERVER = """
def score_order_check(entry, other, capacity, requests):
    if not getattr(entry.policy, "score_group_order", False):
        return dict(passed=True, compiled_enabled=False)
    plan = entry.buffers["group_plan"].cpu().tolist()
    groups, leaves = plan[0], plan[2]
    eligible = (3 * groups == 48 and capacity >= 32 and capacity < 6 * requests
                and 1 + 6 * groups <= 4 * capacity)
    if not eligible:
        return dict(passed=True, compiled_enabled=True, runtime_eligible=False, groups=groups)
    queries, histories = other["query_lengths"], other["history_lengths"]
    assert len(queries) == groups and all(1 <= q <= 6 for q in queries)
    visible = [(h + q) // 4 for h, q in zip(histories, queries)]
    heavy = [q > 3 for q in queries]
    expected = (leaves % 3 == 0 and any(heavy[i] and heavy[i + groups // 2] for i in range(groups // 2))
                and sum(heavy) <= groups // 2 and max(visible) > 0 and max(visible) - min(visible) <= 1)
    storage = entry.buffers["group_offsets"].cpu().tolist()
    base = ((requests + 16) // 16) * 16 + 4 * groups
    actual = storage[base]
    order = storage[base + 1:base + 1 + groups] if actual == 1 else list(range(groups))
    inverse = storage[base + 1 + groups:base + 1 + 2 * groups] if actual == 1 else list(range(groups))
    desired = sorted(range(groups), key=lambda g: (-int(heavy[g]), g)) if expected else list(range(groups))
    passed = actual == int(expected) and order == desired and all(inverse[g] == s for s, g in enumerate(order))
    return dict(passed=passed, compiled_enabled=True, runtime_eligible=True, expected_active=expected,
                actual_active=actual, order=order, inverse=inverse, groups=groups, leaves=leaves)

"""


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/group_state_harness"
    shutil.copytree(here / "runs/measurement_harness", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = (target / "run_pto.py").read_text()
    source = replace_once(source, "or args.state_validation or args.service_validation:", "or args.service_validation:")
    main_fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block = next(n for n in main_fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.state_validation")
    lines = source.splitlines(keepends=True)
    lines[block.lineno - 1 : block.end_lineno] = [
        """    if not args.state_validation or args.case != "h131072_b16_mixed" or args.cpu or args.compile:
        raise ValueError("This harness only validates the fixed C08 SPMD pair; no performance mode")
    case["name"] += "_state_validation"
    case["table_history"] = case["history_label"] + 128
"""
    ]
    source = "".join(lines)
    source = replace_once(
        source,
        '        "device_execution": False,\n',
        '        "device_execution": False,\n        "performance_eligible": False,\n',
    )
    begin = source.index("                for sample in range(3):\n")
    end = source.index('            save("complete")\n', begin) + len('            save("complete")\n')
    source = (
        source[:begin]
        + '            raise AssertionError("State validation must return before any measurement")\n'
        + source[end:]
    )
    assert "pypto.torch.begin_dfx()" not in source and 'report["statistics_us"]' not in source
    (target / "run_pto.py").write_text(source)
    source = (here.parent / "csa_mpmd_tnd_20261021/state_replay.py").read_text()
    source = source.replace('"spmd"', '"reference"').replace('"mpmd"', '"candidate"')
    source = replace_once(source, "def validate(*,", OBSERVER + "def validate(*,")
    source = replace_once(
        source,
        '            {"guards": protected, "same_addresses": same, "generation": protocol, "geometry": geometry},',
        '            {"guards": protected, "same_addresses": same, "generation": protocol, "geometry": geometry,\n'
        '             "score_order": score_order_check(entries[side], other, capacity, requests)},',
    )
    source = replace_once(
        source,
        '                and all(checked["geometry"].values())\n',
        '                and all(checked["geometry"].values())\n                and checked["score_order"]["passed"]\n',
    )
    source = replace_once(
        source,
        '    schedules = [(*schedule, "original") for schedule in schedules]\n',
        """    mixed = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6]
    identity = [q for q in mixed if q > 3] + [q for q in mixed if q <= 3]
    schedules += [
        ("order_enabled", mixed, [history] * 16, None),
        ("order_identity", identity, [history] * 16, None),
        ("order_unequal_history", mixed, [history - 4 * i for i in range(16)], None),
        ("order_reenabled_reverse", mixed, [history] * 16, list(range(15, -1, -1))),
    ]
    schedules = [(*schedule, "original") for schedule in schedules]
    schedules.append(("order_fragmented", mixed, [history] * 16, None, "fragmented_history"))
""",
    )
    (target / "state_replay.py").write_text(source)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    launcher = (here / "launch.sh").read_text()
    launcher = replace_once(
        launcher, 'harness="$here/runs/measurement_harness"', 'harness="$here/runs/group_state_harness"'
    )
    launcher = replace_once(
        launcher, '"$harness/run_pto.py" --family tnd', '"$harness/run_pto.py" --state-validation --family tnd'
    )
    (here / "launch_group_state.sh").write_text(launcher)
    report = dict(
        performance_eligible=False,
        source_validator="../csa_mpmd_tnd_20261021/state_replay.py",
        metadata_cases=15,
        continuous_steps=4,
        additions=[
            "order enabled",
            "identity mapping",
            "unequal histories",
            "reenable with reversed owners",
            "fragmented pages",
        ],
        checks=[
            "same captured addresses",
            "exact reference bytes",
            "guards",
            "generation",
            "geometry",
            "actual forward/inverse map",
        ],
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "group_state_harness.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
