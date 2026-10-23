# SPDX-License-Identifier: Apache-2.0
"""Test fewer G6 Score panels with one QK accumulator and sequential WS Right tiles.

This is independent of scale_gather. It keeps each QK K128 and both WS K192
reductions in the same order, while trading some QK/FIX overlap for fewer panel
iterations. Compiler allocation and events, then full-device checks, decide it.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def assignment(source, name):
    found = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    ]
    assert len(found) == 1, name
    return found[0]


def move_before(source, name, target):
    node = assignment(source, name)
    lines = source.splitlines(keepends=True)
    moved = lines[node.lineno - 1 : node.end_lineno]
    del lines[node.lineno - 1 : node.end_lineno]
    revised = "".join(lines)
    anchor = assignment(revised, target)
    assert node.col_offset == anchor.col_offset
    lines[anchor.lineno - 1 : anchor.lineno - 1] = moved
    return "".join(lines)


def transform(source):
    """Keep one QK Acc object and load each WS Right only at its own use."""
    node = assignment(source, "split_qk_acc1")
    lines = source.splitlines(keepends=True)
    del lines[node.lineno - 1 : node.end_lineno]
    revised = "".join(lines).replace("split_qk_acc1", "split_qk_acc0")
    for name, target in (
        ("split_right0", "split_ws_first"),
        ("split_right1", "split_ws_done"),
        ("split_right1_last", "split_ws_done_last"),
        ("live_half_split_right0", "live_half_split_ws_first"),
    ):
        revised = move_before(revised, name, target)
    revised = replace_once(
        revised,
        "if SPLIT_SCORE_ROWS and balance_leaves and query_group_size == 6 "
        "and qk_panel_cols == 64 and score_tile == 1024:",
        "if SPLIT_SCORE_ROWS and balance_leaves and query_group_size == 6 "
        "and qk_panel_cols == 128 and score_tile == 1024:",
    )
    revised = replace_once(
        revised,
        "            64,  # 此helper固定的数学资源合同；不是T/B/历史实值。",
        "            128,  # N128 experiments use one shared QK accumulator, preserving K128.",
    )
    ast.parse(revised)
    return revised


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="score_n128_single_acc_v3")
    parser.add_argument("--l1-key-prefetch", action="store_true")
    args = parser.parse_args()
    if not args.name.isidentifier():
        raise ValueError("Experiment name must be a simple identifier")
    baseline = here / "runs/baseline"
    expected = json.loads((here / "baseline.json").read_text())["files"]
    source = {p.name: p.read_text() for p in baseline.glob("*.py")}
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()} == expected
    source["indexer_score_stages.py"] = transform(source["indexer_score_stages.py"])
    source["policy.py"] = replace_once(
        source["policy.py"],
        "        qk_panel=64 if group == 6 else 128,",
        "        qk_panel=(128 if long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS\n"
        "                  else 64 if group == 6 else 128),",
    )
    if args.l1_key_prefetch:
        source["policy.py"] = replace_once(
            source["policy.py"],
            '            "TO_L0": True,',
            '            "TO_L0": not (self.split_score_rows and self.qk_panel == 128),',
        )
    for text in source.values():
        ast.parse(text)
    target = here / "runs" / args.name
    target.mkdir(parents=True, exist_ok=False)
    for name, text in source.items():
        (target / name).write_text(text)
    report = dict(
        baseline="baseline.json",
        source=str(target),
        adopted=False,
        device_execution=False,
        changes=["indexer_score_stages.py", "policy.py"],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
        mechanism="N64 to N128 only in existing long-G6 split domain; one QK Acc and sequential WS Right loads",
        preserved=[
            "K128 QK",
            "K192 + K192 WS order",
            "physical leaf boundaries",
            "same Score slot and FIFO event counts",
        ],
        risk="Required FIX-to-MMAD dependency on reused Acc may offset fewer panels; no benefit assumed",
        key_prefetch_to_l1_only=args.l1_key_prefetch,
    )
    with (here / (args.name + ".json")).open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    descriptor = json.loads((here / "descriptors/C04.json").read_text())
    assert descriptor["policy"]["SPLIT_SCORE_ROWS"] and descriptor["policy"]["QK_PANEL"] == 64
    descriptor["policy"]["QK_PANEL"] = 128
    if args.l1_key_prefetch:
        descriptor["policy"]["TO_L0"] = False
    with (here / "descriptors" / ("C04_" + args.name + ".json")).open("x") as f:
        f.write(json.dumps(descriptor, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
