# SPDX-License-Identifier: Apache-2.0
"""检查双QK/WS列池实际地址、数学次数、批量FIX及必要覆盖依赖。"""

import argparse
import ast
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("wave_codegen_helpers", HERE.parent / "wave_current_math/audit_codegen.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    root = HERE / f"cpu_b{args.batch}_v1"
    result = json.loads((root / "result.json").read_text())
    assert result["passed"] and result["single_mix"] and not result["sample_shapes_supplied"]
    assert len(result["symbolic_dimensions"]) == 58 and len(result["compiled_abi"]) == 130
    ir, = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(ir.read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    totals = {}
    for name in ("cube_schedule", "vector_schedule"):
        counts = defaultdict(int)
        for n in functions[name].body:
            if (isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "pl.tile.alloc"):
                counts[ast.unparse(n.value.args[0])] += ast.literal_eval(n.value.args[1])
        assert all(value <= helpers.LIMITS[key] for key, value in counts.items())
        totals[name] = dict(counts)
    records = helpers.annotated(functions["cube_schedule"])
    q0, = [r for r in records if r["name"].startswith("split_qk_acc0_")]
    q1, = [r for r in records if r["name"].startswith("split_qk_acc1_")]
    pool, = [r for r in records if r["name"].startswith("split_ws_pool_")
             and r["operation"].startswith("pl.tile.create(")]
    assert q0["memory"][1:] == [0, 49152]
    assert q1["memory"][1:] == [49152, 49152]
    assert pool["memory"][1:] == [98304, 32768]
    slots = [r for r in records if r["name"].startswith(("split_ws_slot_", "split_ws_last_slot_"))]
    first = [r for r in records if r["name"].startswith("split_ws_first")]
    done = [r for r in records if r["name"].startswith("split_ws_done")]
    assert len(slots) == len(first) == len(done) == 16
    for index, (slot, init, acc) in enumerate(zip(slots, first, done)):
        expected = [98304 + index % 8 * 4096, 4096]
        assert all(r["memory"][1:] == expected for r in (slot, init, acc))
    ir_stores = [n for n in ast.walk(functions["cube_schedule"])
                 if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.store"
                 and ast.unparse(n.args[2]).startswith("indexer_transfer")]
    assert len(ir_stores) == 2
    path = root / "kernel/ptoas/dynamic_csa_full.cpp"
    whole = path.read_text()
    begin = whole.index("AICORE void cube_schedule(")
    end = whole.index("AICORE void vector_schedule(", begin)
    cpp = whole[begin:end]

    def variable(row):
        mark = cpp.index("// pto: %" + row["name"])
        return re.search(r"\b(v\d+) = Tile<TileType::Acc", cpp[mark:]).group(1)

    a, b, p = map(variable, (q0, q1, pool))
    am = list(re.finditer(r"\bTMATMUL\(" + a + r", [^;]+;", cpp))
    bm = list(re.finditer(r"\bTMATMUL\(" + b + r", [^;]+;", cpp))
    af = list(re.finditer(r"\bTINSERT[^\n]+\(v\d+, " + a + r", [^;]+;", cpp))
    bf = list(re.finditer(r"\bTINSERT[^\n]+\(v\d+, " + b + r", [^;]+;", cpp))
    stores = list(re.finditer(r"\bTSTORE\(v\d+, " + p + r"\);", cpp))
    assert all(len(x) == 16 for x in (am, bm, af, bf)) and len(stores) == 2
    score_begin, score_end = am[0].start(), stores[-1].end()
    all_init = list(re.finditer(r"\bTMATMUL\((v\d+), [^;]+;", cpp))
    ws_init = [m for m in all_init if score_begin <= m.start() < score_end and m[1] not in (a, b)]
    all_acc = list(re.finditer(r"\bTMATMUL_ACC\((v\d+), [^;]+;", cpp))
    ws_acc = [m for m in all_acc if score_begin <= m.start() < score_end]
    assert len(ws_init) == len(ws_acc) == 16
    assert len(re.findall(r"\bTINSERT", cpp[score_begin:score_end])) == 32
    events = []
    for index in range(16):
        assert am[index].start() < af[index].start() < bm[index].start() < bf[index].start()
        segments = [("QK0_FIX_to_QK1", af[index].end(), bm[index].start())]
        if index:
            segments.append(("QK1_FIX_to_previous_WS", bf[index].end(), ws_init[index - 1].start()))
        for label, start, stop in segments:
            section = cpp[start:stop]
            sets = re.findall(r"set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", section)
            waits = re.findall(r"wait_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", section)
            assert not set(sets) & set(waits), (index, label, sets, waits)
            events.append({"panel": index, "edge": label, "set": sets, "wait": waits})
    release = re.search(r"set_flag\(PIPE_FIX, PIPE_M, (EVENT_ID\d+)\)", cpp[stores[0].end():])
    assert release
    token = release[1]
    acquire_at = cpp.index(f"wait_flag(PIPE_FIX, PIPE_M, {token})", stores[0].end())
    assert bf[9].end() < acquire_at < ws_init[8].start()
    reference = HERE.parent / "source_after_score_split/indexer_score_stages.py"
    source = HERE / "source_candidate/indexer_score_stages.py"
    def fns(file):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(file.read_text()).body
                if isinstance(n, ast.FunctionDef)}
    old, new = fns(reference), fns(source)
    changed = [name for name in old if old[name] != new[name]]
    assert changed == ["_indexer_score_cube_split_store"]
    report = {
        "passed": True, "ir": str(ir), "cpp": str(path), "dynamic_axes": 58, "root_tensors": 130,
        "single_mix": True, "allocation_totals": totals, "qk_acc": [q0, q1], "ws_pool": pool,
        "slots": slots, "ws_init": first, "ws_acc": done,
        "counts_per_1024": {"qk_mma": 32, "qk_fix": 32, "ws_init": 16, "ws_acc": 16, "ws_store": 2},
        "early_fix_edges": events,
        "pool_overwrite_dependency": {"event": token, "qk9_fix_already_submitted": True,
                                      "acquire_before_ws8_overwrite": True},
        "changed_functions": changed, "device_or_performance_proven": False,
        "note": "机制/地址/事件证据不代替真机精度、完整L1或实际引擎重叠",
    }
    (HERE / f"CODEGEN_B{args.batch}_V1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.batch, "PASS: dual QK 48KiB + WS32KiB; 2 WS stores; required FIX acquire", token)


if __name__ == "__main__":
    main()
