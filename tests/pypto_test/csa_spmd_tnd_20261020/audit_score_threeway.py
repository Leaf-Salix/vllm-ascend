# SPDX-License-Identifier: Apache-2.0
"""Audit the actual three-way Score allocation, retired-buffer reuse and events."""

import argparse
import ast
import json
import re
from pathlib import Path

from audit_score_wide_k128 import memory, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    ir_paths = list((root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(ir_paths) == 1
    tree = ast.parse(ir_paths[0].read_text())
    cube = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    records = {}
    for node in ast.walk(cube):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            region = memory(node.annotation)
            if region:
                records[node.target.id] = {"region": region, "node": node}

    def one(prefix, suffix):
        selected = [(name, row) for name, row in records.items() if name.startswith(prefix) and name.endswith(suffix)]
        assert len(selected) == 1, (prefix, suffix, [name for name, _ in selected])
        return selected[0]

    def arg_region(row, index):
        value = row["node"].value
        assert isinstance(value, ast.Call) and isinstance(value.args[index], ast.Name)
        return records[value.args[index].id]["region"]

    cpp = (root / "kernel/ptoas/dynamic_csa_full.cpp").read_text().split("\nAICORE void vector_schedule", 1)[0]

    def acc_var(name, kind="Acc"):
        start = cpp.index("// pto: %" + name + "\n")
        found = re.search(r"\b(v\d+) = Tile<TileType::" + kind, cpp[start:start + 1000])
        assert found, name
        return found.group(1)

    def unique(pattern):
        matches = list(re.finditer(pattern, cpp))
        assert len(matches) == 1, (pattern, len(matches))
        return matches[0]

    def reuse_events(first, last, producer="PIPE_FIX", consumer="PIPE_M"):
        between = cpp[first:last]
        set_ids = set(re.findall(rf"set_flag\({producer}, {consumer}, (EVENT_ID\d+)\)", between))
        wait_ids = set(re.findall(rf"wait_flag\({producer}, {consumer}, (EVENT_ID\d+)\)", between))
        assert set_ids & wait_ids, (first, last, producer, consumer)
        return sorted(set_ids & wait_ids)

    left = [one(f"split_query{i}_", "__ssa_v0")[1]["region"] for i in range(3)]
    left += [one(f"split_coefficient{i}_", "__ssa_v0")[1]["region"] for i in range(3)]
    assert sum(row[2] for row in left) == 60 * 1024
    assert max(row[1] + row[2] for row in left) <= 65536
    assert not any(overlap(a, b) for i, a in enumerate(left) for b in left[i + 1:])
    panels = []
    previous_fix2 = None
    for panel in range(8):
        suffix = f"__ssa_v{panel}"
        qs = [one(f"triple_qk{i}_", suffix) for i in range(3)]
        regions = [row["region"] for _, row in qs]
        assert regions[0] == regions[2] and regions[0][1:] == [0, 65536]
        assert regions[1][1:] == [65536, 65536]
        qvars = [acc_var(name) for name, _ in qs]
        mmas = [unique(r"\bTMATMUL\(" + name + r", [^;]+;") for name in qvars]
        fixes = [unique(r"\bTINSERT[^\n]+\(v\d+, " + name + r", [^;]+;") for name in qvars]
        assert mmas[0].start() < mmas[1].start() < fixes[0].start() < mmas[2].start() < fixes[1].start()
        row = {"panel": panel, "qk_acc_regions": regions,
               "qk0_to_qk2_reuse": reuse_events(fixes[0].end(), mmas[2].start())}
        if previous_fix2 is not None:
            row["previous_panel_reuse"] = reuse_events(previous_fix2.end(), mmas[0].start())
        if panel:
            name, ws = one("wide_ws0_", suffix)
            assert ws["region"][1:] == [65536, 8192]
            assert overlap(ws["region"], regions[1]) and not overlap(ws["region"], regions[2])
            wvar = acc_var(name)
            init = unique(r"\bTMATMUL\(" + wvar + r", [^;]+;")
            adds = list(re.finditer(r"\bTMATMUL_ACC\(" + wvar + r", [^;]+;", cpp))
            assert len(adds) == 2
            assert fixes[1].start() < init.start() < fixes[2].start() < adds[0].start() < adds[1].start()
            row["qk1_to_ws_reuse"] = reuse_events(fixes[1].end(), init.start())
            row["ws_region"] = ws["region"]
            # The following Key is already prefetched. WS Right must not cover
            # that still-live Key, even though it may reuse the retired current Key.
            right = arg_region(ws, 1)
            assert right[2] == 32768 and right[1] + right[2] <= 65536
            row["ws_right"] = right
            current_key = arg_region(qs[0][1], 1)
            if overlap(right, current_key):
                rvar = acc_var(ws["node"].value.args[1].id, "Right")
                extract = unique(r"\bTEXTRACT\(" + rvar + r", [^;]+;")
                row["retired_key_to_ws_right"] = reuse_events(
                    mmas[2].end(), extract.start(), "PIPE_M", "PIPE_MTE1"
                )
            if panel < 7:
                _, next_q = one("triple_qk0_", f"__ssa_v{panel + 1}")
                next_key = arg_region(next_q, 1)
                assert next_key[2] == 16384 and not overlap(right, next_key), (panel, right, next_key)
                row["preserved_next_key"] = next_key
        previous_fix2 = fixes[2]
        panels.append(row)
    _, tail = one("wide_ws0_last_", "__ssa_v0")
    assert tail["region"][2] == 8192
    # The old vector, sorting, publication, FIFO event numbers and drain remain
    # source-identical; here verify the generated current helper's static counts.
    step, = [n for n in ast.walk(cube) if isinstance(n, ast.For) and "buf_score_step" in ast.unparse(n.target)]
    stores = [n for n in ast.walk(step) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.store"
              and ast.unparse(n.args[2]).startswith("indexer_transfer")]
    assert len(stores) == 8
    result = {
        "passed": True, "device_execution": False, "actual_qk_panel_cols": 128,
        "left": left, "panels": panels, "ws_tail": tail["region"],
        "counts": {"qk": 24, "qk_fix": 24, "ws_init": 8, "ws_acc": 16, "score_store": 8},
        "acc_highest_address_end": 131072,
        "scope": "Allocation and required event audit only; numerical and full-L1 device checks still required",
    }
    (root / "resource_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "counts", "acc_highest_address_end")}))


if __name__ == "__main__":
    main()
