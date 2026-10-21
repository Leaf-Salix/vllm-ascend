# SPDX-License-Identifier: Apache-2.0
"""Port live-half math and optionally rotate leaves with matching SPMD writer waits."""

import ast
import copy
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

from prepare_control_candidates import replace_once


class FirstHalfOnly(ast.NodeTransformer):
    """Mechanism adapted from the other session's prepare_score_live_half.py.

    Its MPMD measurements are not SPMD evidence. Keep the six-row FIFO and
    every READY/ACK/drain call, while using fixed M192 on live<=3 groups.
    """

    second_half = {
        "split_query1",
        "split_coefficient1",
        "split_qk_acc1",
        "split_qk1",
        "split_previous1",
        "split_current1",
        "split_right1",
        "split_right1_last",
        "split_ws_done",
        "split_ws_done_last",
    }

    def __init__(self):
        self.removed = []

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in self.second_half:
                self.removed.append(node.targets[0].id)
                return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        replacement = {"split_ws_done": "split_ws_first", "split_ws_done_last": "split_ws_first_last"}.get(node.id)
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node) if replacement else node

    def visit_BinOp(self, node):
        if ast.unparse(node) == "query_group_size * IDX_N_HEADS":
            return ast.copy_location(ast.Name(id="SPLIT_QK_ROWS", ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if ast.unparse(node.func) == "pl.unroll" and len(node.args) == 1:
            if ast.unparse(node.args[0]) == "query_group_size":
                node.args[0] = ast.parse("SPLIT_QK_ROWS // IDX_N_HEADS", mode="eval").body
        return self.generic_visit(node)


def calls(body, name):
    return [
        ast.dump(n)
        for n in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == name
    ]


def live_half_source(source):
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    valid = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "work_valid > 0"]
    assert len(valid) == 1
    valid = valid[0]
    full = copy.deepcopy(valid.body)
    transform = FirstHalfOnly()
    half_module = transform.visit(ast.Module(body=copy.deepcopy(full), type_ignores=[]))
    assert set(transform.removed) == transform.second_half
    for name in ("pl.system.sync_set", "pl.system.sync_wait"):
        assert calls(full, name) == calls(half_module.body, name), name
    assert len(calls(full, "pl.store")) == len(calls(half_module.body, "pl.store")) == 2
    assert not any(isinstance(n, ast.Name) and n.id in transform.second_half for n in ast.walk(half_module))
    parameters = {arg.arg for arg in fn.args.args}
    local_names = {
        n.id
        for n in ast.walk(half_module)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in parameters
    }
    for node in ast.walk(half_module):
        if isinstance(node, ast.Name) and node.id in local_names:
            node.id = "live_half_" + node.id
    valid.body = [
        ast.If(
            test=ast.parse("buf_live <= SPLIT_QK_ROWS // IDX_N_HEADS", mode="eval").body,
            body=half_module.body,
            orelse=full,
        )
    ]
    assert ast.dump(ast.Module(body=valid.body[0].orelse, type_ignores=[])) == ast.dump(
        ast.Module(body=full, type_ignores=[])
    )
    lines = source.splitlines(keepends=True)
    comment = "# live<=3: fixed M192, original six-row FIFO and both AIV ACKs; full branch preserves v122 math.\n"
    lines[valid.lineno - 1 : valid.end_lineno] = [
        textwrap.indent(comment + ast.unparse(ast.fix_missing_locations(valid)), " " * valid.col_offset) + "\n"
    ]
    return "".join(lines)


def rotate_sources(group_source, score_source, dependency_source):
    group_source = replace_once(
        group_source,
        "    leaves: pl.Scalar[pl.INDEX],\n",
        "    leaves: pl.Scalar[pl.INDEX],\n    GROUP_SIZE: pl.constexpr,\n",
    )
    group_source = replace_once(
        group_source,
        "            group = item % groups\n            leaf = item // groups\n",
        "            leaf = item // groups\n"
        "            group = item % groups\n"
        "            if GROUP_SIZE == 6:\n"
        "                group = (group + leaf) % groups\n",
    )
    before = "indexer_work_item(buf_item, groups, buf_max_leaves)"
    assert score_source.count(before) == 3
    score_source = score_source.replace(before, "indexer_work_item(buf_item, groups, buf_max_leaves, query_group_size)")
    dependency_source = replace_once(
        dependency_source,
        "            item = leaf * groups + group\n",
        "            ordinal = group\n"
        "            if QUERY_GROUP == 6:\n"
        "                ordinal = (group + groups - leaf % groups) % groups\n"
        "            item = leaf * groups + ordinal\n",
    )
    return group_source, score_source, dependency_source


def main():
    here = Path(__file__).resolve().parent
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    score = live_half_source((reference / "indexer_score_stages.py").read_text())
    report = {"reference_kind": "frozen_tnd_v122", "adopted": False, "device_execution": False, "variants": {}}
    for name in ("live_half", "live_half_rotate"):
        target = here / "runs" / name
        shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        revised = {"indexer_score_stages.py": score}
        if name == "live_half_rotate":
            group, revised_score, dependency = rotate_sources(
                (reference / "indexer_group_stages.py").read_text(),
                score,
                (reference / "attention_dependencies.py").read_text(),
            )
            revised.update(
                {
                    "indexer_group_stages.py": group,
                    "indexer_score_stages.py": revised_score,
                    "attention_dependencies.py": dependency,
                }
            )
        for filename, source in revised.items():
            ast.parse(source)
            (target / filename).write_text(source)
        report["variants"][name] = {
            "source": str(target),
            "changed_files": list(revised),
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        }
    old = here.parent / "csa_spmd_tnd_20261020"
    mixed = old / "runs/requested_s6_mixed_b16_20261022_v124/mixed_s1_s6/buffer_descriptors.json"
    shutil.copyfile(mixed, here / "descriptors/mixed_h131072_b16_t56.json")
    with (here / "mixed_candidates.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prepared": list(report["variants"]), "device_execution": False}))


if __name__ == "__main__":
    main()
