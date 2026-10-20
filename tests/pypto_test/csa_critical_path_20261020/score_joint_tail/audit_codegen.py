# SPDX-License-Identifier: Apache-2.0
"""审查短尾与整块的实际数学、FIFO宽度、资源和生成代码；不代替设备验证。"""

import argparse
import ast
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True)
    args = parser.parse_args()
    root = HERE / args.cpu
    result = json.loads((root / "result.json").read_text())
    assert result["passed"] and result["single_mix"] and not result["sample_shapes_supplied"]
    assert len(result["symbolic_dimensions"]) == 58 and len(result["compiled_abi"]) == 130
    spec = importlib.util.spec_from_file_location("joint_helpers", HERE.parent / "wave_current_math/audit_codegen.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    ir, = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    functions = {n.name: n for n in ast.walk(ast.parse(ir.read_text())) if isinstance(n, ast.FunctionDef)}
    allocations = {}
    for name in ("cube_schedule", "vector_schedule"):
        total = defaultdict(int)
        for n in functions[name].body:
            if isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "pl.tile.alloc":
                total[ast.unparse(n.value.args[0])] += ast.literal_eval(n.value.args[1])
        assert all(v <= helper.LIMITS[k] for k, v in total.items())
        allocations[name] = dict(total)
    records = helper.annotated(functions["cube_schedule"])
    def one(prefix):
        rows = [r for r in records if r["name"].startswith(prefix)]
        assert len(rows) == 1, (prefix, len(rows))
        return rows[0]
    q0 = one("split_qk_acc0_")
    q1 = one("split_qk_acc1_")
    assert q0["memory"][1:] == [0, 49152] and q1["memory"][1:] == [49152, 49152]
    for prefix in ("tail_ws_slot_", "tail_ws_first_", "tail_ws_done_", "tail_ws_store_"):
        assert one(prefix)["memory"][1:] == [98304, 4096]
    branches = [n for n in ast.walk(functions["cube_schedule"]) if isinstance(n, ast.If)
                and any(isinstance(x, ast.AnnAssign) and ast.unparse(x.target).startswith("tail_qk0_") for x in n.body)]
    branch, = branches
    counts = {}
    for name, statements in (("short", branch.body), ("full", branch.orelse)):
        block = ast.Module(body=statements, type_ignores=[])
        mm = calls(block, "pl.tile.matmul_acc")
        stores = calls(block, "pl.tile.store")
        counts[name] = {"matmuls": len(mm), "fifo_stores": len(stores)}
        assert len(mm) == (4 if name == "short" else 64)
        assert len(stores) == (1 if name == "short" else 2)
        assert all(ast.unparse(n.args[2]).startswith("indexer_transfer") for n in stores)
    vrecords = helper.annotated(functions["vector_schedule"])
    loads = [r for r in vrecords if r["name"].startswith("tail_single_scores_")]
    assert len(loads) == 1 and "[3, 64], [3, 64]" in loads[0]["operation"]
    vbranches = [n for n in ast.walk(functions["vector_schedule"]) if isinstance(n, ast.If)
                 and any(isinstance(x, ast.AnnAssign) and ast.unparse(x.target).startswith("tail_single_scores_") for x in n.body)]
    vbranch, = vbranches
    for statements in (vbranch.body, vbranch.orelse):
        block = ast.Module(body=statements, type_ignores=[])
        assert len(calls(block, "pl.system.sync_wait")) == 1
        assert len(calls(block, "pl.system.sync_set")) == 1
    cpp_file = root / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_file.read_text()
    cube = cpp[cpp.index("AICORE void cube_schedule("):cpp.index("AICORE void vector_schedule(")]
    def variable(row):
        at = cube.index("// pto: %" + row["name"])
        return re.search(r"\b(v\d+) = Tile<TileType::Acc", cube[at:]).group(1)
    a, b = variable(q0), variable(q1)
    for var in (a, b):
        assert len(re.findall(r"\bTMATMUL\(" + var + r",", cube)) == 17
        assert len(re.findall(r"\bTINSERT[^\n]+\(v\d+, " + var + r",", cube)) == 17
    source = HERE / "source_candidate"
    original = HERE.parent / "score_split_pool/source_candidate"
    changed = [p.name for p in source.glob("*.py") if p.read_bytes() != (original / p.name).read_bytes()]
    assert sorted(changed) == ["entry.py", "indexer_score_stages.py", "vector_schedule.py"]
    marker = "                    if single_step % 2 == 1 or single_step + 1 == single_steps:"
    assert (source / "indexer_score_stages.py").read_text().split(marker)[1] == (original / "indexer_score_stages.py").read_text().split(marker)[1]
    report = {"passed": True, "ir": str(ir), "cpp": str(cpp_file), "dynamic_axes": 58,
              "root_tensors": 130, "single_mix": True, "allocation_totals": allocations,
              "branch_counts": counts, "short_condition": ast.unparse(branch.test),
              "vector_short_condition": ast.unparse(vbranch.test), "tail_fifo_load": loads,
              "qk_acc": [q0,q1], "tail_ws_store": one("tail_ws_store_"),
              "cpp_qk_mma_and_fix_per_half": 17, "changed_files_vs_pool": sorted(changed),
              "sorting_source_unchanged": True, "one_ready_ack_per_branch": True,
              "device_or_performance_proven": False}
    (HERE / (args.cpu.upper() + "_AUDIT.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
