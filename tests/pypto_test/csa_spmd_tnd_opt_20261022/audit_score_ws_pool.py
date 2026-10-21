# SPDX-License-Identifier: Apache-2.0
"""Audit N64 QK/WS allocations, write counts and pool retirement dependencies."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from audit_score_wide_k128 import memory, overlap  # noqa: E402


def audit(root):
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    (ir_path,) = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(ir_path.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("cube_schedule", "score_producer")
    )
    records = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            region = memory(node.annotation)
            if region:
                records[node.target.id] = dict(region=region, line=node.lineno)

    def selected(prefix):
        return sorted(((n, r) for n, r in records.items() if n.startswith(prefix)), key=lambda v: v[1]["line"])

    def initial(prefix):
        result = [(n, r) for n, r in selected(prefix) if n.endswith("__ssa_v0")]
        assert len(result) == 1, (prefix, result)
        return result[0]

    q0n, q0r = initial("split_qk_acc0_")
    q1n, q1r = initial("split_qk_acc1_")
    pn, pr = initial("split_ws_pool_")
    regions = [r["region"] for r in (q0r, q1r, pr)]
    assert [r[2] for r in regions] == [49152, 49152, 32768]
    assert max(r[1] + r[2] for r in regions) <= 131072
    assert not any(overlap(a, b) for i, a in enumerate(regions) for b in regions[i + 1 :])
    slots = selected("split_ws_slot_") + selected("split_ws_last_slot_")
    assert len(slots) == 16
    assert [r["region"][1] for _, r in slots] == [pr["region"][1] + (i % 8) * 4096 for i in range(16)]
    assert all(r["region"][2] == 4096 for _, r in slots)

    cpp_path = root / "kernel/ptoas" / ("dynamic_csa_full.cpp" if fn.name == "cube_schedule" else "score_producer.cpp")
    cpp = cpp_path.read_text().split("\nAICORE void vector_schedule", 1)[0]

    def variable(name):
        start = cpp.index("// pto: %" + name + "\n")
        found = re.search(r"\b(v\d+) = Tile<TileType::Acc", cpp[start : start + 1000])
        assert found, name
        return found.group(1)

    q0, q1, pool = map(variable, (q0n, q1n, pn))
    qm = [list(re.finditer(r"\bTMATMUL\(" + v + r", [^;]+;", cpp)) for v in (q0, q1)]
    fixes = [list(re.finditer(r"\bTINSERT[^\n]+\(v\d+, " + v + r", [^;]+;", cpp)) for v in (q0, q1)]
    stores = list(re.finditer(r"\bTSTORE\(v\d+, " + pool + r"\);", cpp))
    assert [len(v) for v in qm + fixes] == [16] * 4
    assert len(stores) == 2
    block = cpp[qm[0][0].start() : stores[-1].end()]
    mm = list(re.finditer(r"\bTMATMUL\((v\d+), [^;]+;", cpp))
    ws = [m for m in mm if m.group(1) not in (q0, q1) and qm[0][0].start() < m.start() < stores[-1].start()]
    assert len(ws) == 16
    assert len(re.findall(r"\bTMATMUL_ACC\(", block)) == 16
    assert len(re.findall(r"\bTINSERT[<\(]", block)) == 32, "No Acc copies or extra WS FIX allowed"
    assert len(re.findall(r"\bTSTORE\(", block)) == 2

    def dependency(start, end):
        between = cpp[start:end]
        sets = set(re.findall(r"set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", between))
        waits = set(re.findall(r"wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", between))
        assert sets & waits, (start, end)
        return sorted(sets & waits)

    # Retire the first 512-column store before reusing the same pool at WS8.
    assert ws[7].end() < stores[0].start() < qm[0][9].start() < ws[8].start()
    middle = dependency(stores[0].end(), ws[8].start())
    # The final store is in FIX before the next iteration's QK0 FIX. That FIX's
    # completion is acquired before QK0(panel1), which precedes WS0. Thus the
    # loop-carried pool reuse is protected by FIX FIFO order + FIX->M, even if
    # there is no redundant direct flag immediately after the final TSTORE.
    assert fixes[0][0].end() < qm[0][1].start() < ws[0].start()
    loop = dependency(fixes[0][0].end(), qm[0][1].start())
    assert stores[-1].end() > ws[-1].end()
    return {
        "compile_dir": str(root),
        "passed": True,
        "ir": str(ir_path),
        "cpp": str(cpp_path),
        "qk0": regions[0],
        "qk1": regions[1],
        "ws_pool": regions[2],
        "ws_slots": [r["region"] for _, r in slots],
        "counts_per_1024": dict(qk_mma=32, qk_fix=32, ws_init=16, ws_acc=16, ws_store=2),
        "middle_pool_reuse_fix_to_m": middle,
        "loop_pool_reuse": dict(
            path="last store -> FIX FIFO -> next QK0 FIX -> FIX/M -> QK0 panel1 -> WS0", events=loop
        ),
        "limitations": "Static allocation/order/dependency checks only; no precision or measured overlap claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": True, "device_execution": False, "cases": [audit(p.resolve()) for p in args.compile_dirs]}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "compile_cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
