# SPDX-License-Identifier: Apache-2.0
"""Audit two WS Right slots, retired Key reuse and unchanged matrix ordering."""

import argparse
import ast
import json
import re
from pathlib import Path

from audit_score_wide_k128 import memory, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    path, = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(path.read_text())
    cube = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    rows = {}
    for node in ast.walk(cube):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            region = memory(node.annotation)
            if region:
                rows[node.target.id] = {"node": node, "region": region, "name": node.target.id}

    def named(prefix):
        return sorted((r for n, r in rows.items() if n.startswith(prefix)), key=lambda r: r["node"].lineno)

    def operand(row, index):
        return rows[row["node"].value.args[index].id]

    cpp = (root / "kernel/ptoas/dynamic_csa_full.cpp").read_text().split("\nAICORE void vector_schedule", 1)[0]

    def variable(row, kind):
        start = cpp.index("// pto: %" + row["name"] + "\n")
        match = re.search(r"\b(v\d+) = Tile<TileType::" + kind, cpp[start:start + 1000])
        assert match, row["name"]
        return match.group(1)

    def unique(pattern):
        found = list(re.finditer(pattern, cpp))
        assert len(found) == 1, (pattern, len(found))
        return found[0]

    def matrix(row):
        v = variable(row, "Acc")
        return unique(r"\bTMATMUL\(" + v + r", [^;]+;"), v

    def reuse(begin, end, source, destination):
        between = cpp[begin:end]
        set_ids = set(re.findall(rf"set_flag\({source}, {destination}, (EVENT_ID\d+)\)", between))
        wait_ids = set(re.findall(rf"wait_flag\({source}, {destination}, (EVENT_ID\d+)\)", between))
        assert set_ids & wait_ids, (begin, end, source, destination)
        return sorted(set_ids & wait_ids)

    qs = [named(f"triple_qk{i}_") for i in range(3)]
    ws = [named(f"wide_ws{i}_") for i in range(3)]
    assert all(len(v) == 8 for v in qs + ws)
    panels = []
    for panel in range(8):
        q = [v[panel] for v in qs]
        assert [r["region"][1:] for r in q] == [[0, 65536], [65536, 65536], [0, 65536]]
        mm_and_vars = [matrix(row) for row in q]
        mm = [item[0] for item in mm_and_vars]
        fixes = [unique(r"\bTINSERT[^\n]+\(v\d+, " + v + r", [^;]+;") for _, v in mm_and_vars]
        assert mm[0].start() < mm[1].start() < fixes[0].start() < mm[2].start() < fixes[1].start()
        report = {"panel": panel, "qk_acc": [r["region"] for r in q],
                  "qk_reuse": reuse(fixes[0].end(), mm[2].start(), "PIPE_FIX", "PIPE_M")}
        if panel:
            w = [v[panel - 1] for v in ws]
            rights = [operand(w[0], 1), operand(w[1], 2), operand(w[2], 2)]
            regions = [r["region"] for r in rights]
            assert [r[1:] for r in regions] == [[0, 32768], [32768, 32768], [0, 32768]]
            key = operand(q[0], 1)["region"]
            assert key[1:] == [32768, 16384]
            assert not overlap(key, regions[0]) and overlap(key, regions[1])
            extracts = [unique(r"\bTEXTRACT\(" + variable(r, "Right") + r", [^;]+;") for r in rights]
            init, wvar = matrix(w[0])
            adds = list(re.finditer(r"\bTMATMUL_ACC\(" + wvar + r", [^;]+;", cpp))
            assert len(adds) == 2
            assert extracts[0].start() < mm[0].start()
            assert mm[2].start() < extracts[1].start() < init.start() < fixes[2].start()
            assert init.start() < extracts[2].start() < adds[0].start() < adds[1].start()
            assert w[0]["region"][1:] == [65536, 8192]
            report.update(
                ws_right=regions, key_right=key,
                qk1_to_ws=reuse(fixes[1].end(), init.start(), "PIPE_FIX", "PIPE_M"),
                key_to_ws_right=reuse(mm[2].end(), extracts[1].start(), "PIPE_M", "PIPE_MTE1"),
                ws_right_reuse=reuse(init.end(), extracts[2].start(), "PIPE_M", "PIPE_MTE1"),
            )
        panels.append(report)
    # The final panel drains the same three K128 WS products, also with two slots.
    tail = [v[-1] for v in ws]
    tail_rights = [operand(tail[0], 1), operand(tail[1], 2), operand(tail[2], 2)]
    assert [r["region"][1:] for r in tail_rights] == [[0, 32768], [32768, 32768], [0, 32768]]
    step, = [n for n in ast.walk(cube) if isinstance(n, ast.For) and "buf_score_step" in ast.unparse(n.target)]
    stores = [n for n in ast.walk(step) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.store"
              and ast.unparse(n.args[2]).startswith("indexer_transfer")]
    assert len(stores) == 8
    result = {
        "passed": True, "device_execution": False, "panels": panels,
        "counts": {"qk": 24, "qk_fix": 24, "ws_init": 8, "ws_acc": 16, "score_store": 8},
        "scope": "Actual regions and mandatory reuse events; not measured overlap or performance",
    }
    (root / "resource_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "counts": result["counts"]}))


if __name__ == "__main__":
    main()
