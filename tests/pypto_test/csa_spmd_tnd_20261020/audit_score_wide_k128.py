# SPDX-License-Identifier: Apache-2.0
"""Check the generated N128/K128 Score resource layout."""

import argparse
import ast
import json
import re
from pathlib import Path


def memory(annotation):
    refs = [n for n in ast.walk(annotation) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.MemRef"]
    if len(refs) != 1:
        return None
    root, offset, size = refs[0].args
    value = ast.literal_eval(offset.args[0]) if isinstance(offset, ast.Call) else ast.literal_eval(offset)
    return [ast.unparse(root), value, ast.literal_eval(size)]


def overlap(a, b):
    return a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dir", type=Path)
    args = parser.parse_args()
    root = args.compile_dir
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    paths = list((root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(paths) == 1
    tree = ast.parse(paths[0].read_text())
    cube = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    records = {}
    for node in ast.walk(cube):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            region = memory(node.annotation)
            if region:
                records[node.target.id] = {"region": region, "operation": ast.unparse(node.value), "line": node.lineno}

    def one(prefix):
        values = [r["region"] for n, r in records.items() if n.startswith(prefix) and n.endswith("__ssa_v0")]
        assert len(values) == 1, (prefix, values)
        return values[0]

    qk, ws = one("split_qk_acc_"), one("split_ws_acc_")
    assert qk[2] == 98304 and ws[2] == 8192 and not overlap(qk, ws)
    left = [one(f"split_query{i}_") for i in (0, 1)] + [one(f"split_coefficient{i}_") for i in (0, 1, 2)]
    assert sorted(r[2] for r in left) == [4096, 4096, 4096, 24576, 24576]
    assert max(r[1] + r[2] for r in left) <= 65536
    assert not any(overlap(a, b) for i, a in enumerate(left) for b in left[i + 1 :])
    rights = [r["region"] for n, r in records.items() if re.match(r"wide_right[012](?:_last)?_", n)]
    right_regions = sorted({tuple(r) for r in rights})
    assert len(rights) == 24 and len(right_regions) == 1 and right_regions[0][2] == 32768, right_regions
    keys = [r["region"] for n, r in sorted(records.items(), key=lambda item: item[1]["line"])
            if n.startswith(("buf_current_key_", "buf_prefetched_key_")) and "pl.tile.extract" in r["operation"]]
    key_regions = sorted({tuple(r) for r in keys})
    assert len(keys) == 8 and all(r[2] == 16384 for r in keys), keys
    # The priming panel has no previous WS. Its key may occupy the later WS
    # region, but all subsequent prefetched keys must use two disjoint slots.
    steady_key_regions = sorted({tuple(r) for r in keys[1:]})
    assert len(steady_key_regions) == 2, steady_key_regions
    assert not overlap(*steady_key_regions)
    assert not overlap(keys[0], keys[1])
    assert all(not overlap(k, right_regions[0]) for k in steady_key_regions)
    assert max(r[1] + r[2] for r in (*key_regions, *right_regions)) <= 65536
    step, = [n for n in ast.walk(cube) if isinstance(n, ast.For) and "buf_score_step" in ast.unparse(n.target)]
    calls = [n for n in ast.walk(step) if isinstance(n, ast.Call)]
    qk_calls = [n for n in calls if ast.unparse(n.func) == "pl.tile.matmul_acc"
                and ast.unparse(n.args[0]).startswith("split_qk_acc")]
    ws_calls = [n for n in calls if ast.unparse(n.func) == "pl.tile.matmul_acc"
                and ast.unparse(n.args[0]).startswith(("split_ws_acc", "wide_ws"))]
    stores = [n for n in calls if ast.unparse(n.func) == "pl.tile.store"
              and ast.unparse(n.args[2]).startswith("indexer_transfer")]
    assert (len(qk_calls), len(ws_calls), len(stores)) == (16, 24, 8)
    cpp = (root / "kernel/ptoas/dynamic_csa_full.cpp").read_text()

    def acc_var(prefix):
        start = cpp.index("// pto: %" + prefix)
        return re.search(r"\b(v\d+) = Tile<TileType::Acc", cpp[start:]).group(1)

    qvar, wvar = acc_var("split_qk_acc_"), acc_var("split_ws_acc_")
    qmm = list(re.finditer(r"\bTMATMUL\(" + qvar + r", [^;]+;", cpp))
    qfix = list(re.finditer(r"\bTINSERT[^\n]+\(v\d+, " + qvar + r", [^;]+;", cpp))
    winit = list(re.finditer(r"\bTMATMUL\(" + wvar + r", [^;]+;", cpp))
    wacc = list(re.finditer(r"\bTMATMUL_ACC\(" + wvar + r", [^;]+;", cpp))
    assert (len(qmm), len(qfix), len(winit), len(wacc)) == (16, 16, 8, 16)
    assert qmm[1].end() < winit[0].start()
    reuse = []
    for index in range(8):
        q0, q1, f0, f1 = qmm[2 * index], qmm[2 * index + 1], qfix[2 * index], qfix[2 * index + 1]
        assert q0.start() < f0.start() < q1.start() < f1.start()
        between = cpp[f0.end() : q1.start()]
        sets = set(re.findall(r"set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", between))
        waits = set(re.findall(r"wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", between))
        assert sets & waits, (index, "missing QK Acc reuse dependency")
        reuse.append(sorted(sets & waits))
    result = dict(
        passed=True, device_execution=False, actual_panel_cols=128, ir=str(paths[0]),
        qk_acc=qk, ws_acc=ws, left=left, ws_right=right_regions, keys_right=key_regions,
        priming_key=keys[0], steady_keys=steady_key_regions,
        counts={"qk": 16, "qk_fix": 16, "ws_init": 8, "ws_acc": 16, "store": 8},
        qk_acc_reuse_events=reuse,
        limitation="Resource/operation/dependency audit only; device precision and L1 performance are still required",
    )
    (root / "resource_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
