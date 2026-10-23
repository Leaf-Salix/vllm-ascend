# SPDX-License-Identifier: Apache-2.0
"""Compile group ordering only in the mixed-capacity long-G6 domain."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once


def wire(source, names):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    found = set()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start, stop = offsets[node.lineno - 1], offsets[node.body[0].lineno - 1]
            pos = start + source[start:stop].rindex("):")
            edits.append((pos, pos, "    SCORE_GROUP_ORDER: pl.constexpr = False,\n"))
            found.add(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            node.keywords.append(ast.keyword("SCORE_GROUP_ORDER", ast.Name("SCORE_GROUP_ORDER", ast.Load())))
            start = offsets[node.lineno - 1] + node.col_offset
            stop = offsets[node.end_lineno - 1] + node.end_col_offset
            edits.append((start, stop, ast.unparse(node)))
    for start, stop, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[stop:]
    ast.parse(source)
    return source, found


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/score_group_order_v2"
    expected = json.loads((here / "score_group_order_v2.json").read_text())["files"]
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    names = {
        "decode_csa_spmd",
        "_csa_core_group",
        "cube_schedule",
        "vector_schedule",
        "indexer_score_cube_stage",
        "_indexer_score_cube_original",
        "_indexer_score_cube_split_store",
        "indexer_score_vector_lane_stage",
        "indexer_group_plan_stage",
        "indexer_pair_publish_stage",
        "wait_indexer_query_roots",
    }
    found = set()
    for file in (
        "entry.py",
        "cube_schedule.py",
        "vector_schedule.py",
        "indexer_score_stages.py",
        "indexer_group_stages.py",
        "indexer_publish_stages.py",
        "attention_dependencies.py",
    ):
        source[file], present = wire(source[file], names)
        found.update(present)
    assert found == names
    for file, old in (
        ("indexer_group_stages.py", "if BALANCED and GROUP_SIZE == 6:"),
        ("attention_dependencies.py", "if BALANCED and QUERY_GROUP == 6:"),
    ):
        source[file] = replace_once(source[file], old, old[:-1] + " and SCORE_GROUP_ORDER:")
    old = "if balance_leaves and query_group_size == 6:"
    assert source["indexer_score_stages.py"].count(old) == 3
    source["indexer_score_stages.py"] = source["indexer_score_stages.py"].replace(
        old, old[:-1] + " and SCORE_GROUP_ORDER:"
    )
    source["policy.py"] = replace_once(
        source["policy.py"],
        "    prefix_branch_groups: int = 4\n",
        "    prefix_branch_groups: int = 4\n    score_group_order: bool = False\n",
    )
    source["policy.py"] = replace_once(
        source["policy.py"],
        '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n',
        '            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,\n'
        '            "SCORE_GROUP_ORDER": self.score_group_order,\n',
    )
    source["policy.py"] = replace_once(
        source["policy.py"],
        "        prefix_branch_groups=4,\n",
        "        prefix_branch_groups=4,\n"
        "        score_group_order=(long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS\n"
        "                           and tokens < requests * group),\n",
    )
    target = here / "runs/score_group_order_v3"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = dict(
        source=str(target),
        baseline="baseline.json",
        parent="score_group_order_v2",
        adopted=False,
        device_execution=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        mechanism=(
            "Existing Host capacity selects compilation only; "
            "actual groups, live halves and history checks remain on device"
        ),
        required=(
            "Prove enabled C08 matches measured v2 and disabled seven cases match their historical generated code; "
            "validate metadata replay"
        ),
    )
    with (here / "score_group_order_v3.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    baseline = json.loads((here / "baseline.json").read_text())
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    for key, row in baseline["cases"].items():
        measured = json.loads(Path(row["source_report"]).read_text())
        case = measured["case"]
        policy = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        assert policy.pop("SCORE_GROUP_ORDER") == (key == "C08")
        assert policy == measured["policies"]["spmd"]
        policy["SCORE_GROUP_ORDER"] = key == "C08"
        descriptor = dict(buffers=measured["descriptors"]["spmd"], policy=policy)
        with (here / "descriptors" / f"{key}_score_group_order_v3.json").open("x") as stream:
            stream.write(json.dumps(descriptor, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
