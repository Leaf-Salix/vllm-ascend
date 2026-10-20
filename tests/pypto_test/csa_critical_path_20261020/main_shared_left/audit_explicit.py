# SPDX-License-Identifier: Apache-2.0
"""审查完整编译中的主压缩双Right、共享Left及动态compact累加合同。"""

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wave_current_math"))
from audit_codegen import LIMITS, annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    root = HERE / f"cpu_b{args.batch}_v2"
    result = json.loads((root / "result.json").read_text())
    assert result["passed"] and result["single_mix"] and not result["sample_shapes_supplied"]
    assert len(result["symbolic_dimensions"]) == 58 and len(result["compiled_abi"]) == 130
    ir, = (root / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
    tree = ast.parse(ir.read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    totals = {}
    for name in ("cube_schedule", "vector_schedule"):
        counts = defaultdict(int)
        for node in functions[name].body:
            if (isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "pl.tile.alloc"):
                counts[ast.unparse(node.value.args[0])] += ast.literal_eval(node.value.args[1])
        assert all(value <= LIMITS[key] for key, value in counts.items()), counts
        totals[name] = dict(counts)
    cube = annotated(functions["cube_schedule"])
    ann = {n.target.id: ast.unparse(n.annotation) for n in ast.walk(functions["cube_schedule"])
           if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    left = [r for r in cube if r["name"].startswith(("left0_", "left1_"))]
    right = [r for r in cube if r["name"].startswith(("kv_right0_", "kv_right1_", "gate_right0_", "gate_right1_"))]
    mat = [r for r in cube if r["name"].startswith(("x_l1_", "kv_l1_", "gate_l1_"))
           and r["operation"].startswith("pl.tile.load(")]
    assert len(left) == 4 and all(r["memory"][2] == 32768 for r in left)
    assert [r["memory"][1] for r in left] == [0, 32768, 0, 32768]
    assert len(right) == 8 and all(r["memory"][2] == 32768 for r in right)
    assert [r["memory"][1] for r in right] == [0, 32768, 0, 32768] * 2
    assert len(mat) == 6 and all(r["memory"][2] == 65536 for r in mat)
    assert len({r["memory"][1] for r in mat}) == 6
    assert max(r["line"] for r in mat) < min(r["line"] for r in left)
    operations = []
    usage = defaultdict(list)
    for row in cube:
        if not row["operation"].startswith("pl.tile.matmul_acc("):
            continue
        call = ast.parse(row["operation"], mode="eval").body
        lhs = ast.unparse(call.args[1])
        if lhs not in {r["name"] for r in left}:
            continue
        rhs = ast.unparse(call.args[2])
        usage[lhs].append(row["name"])
        assert "compact=pl.CompactMode.normal" in ann[row["name"]]
        assert "x_rows_" in ann[row["name"]]
        operations.append({**row, "lhs": lhs, "rhs": rhs})
    assert len(operations) == 8 and all(len(v) == 2 for v in usage.values())
    expected_lhs = [left[i]["name"] for i in (0, 0, 1, 1, 2, 2, 3, 3)]
    assert [r["lhs"] for r in operations] == expected_lhs
    assert [r["rhs"] for r in operations] == [r["name"] for r in right]
    # 第二份KV Right在第一份KV发出后恢复；Gate1在Gate0发出后恢复。
    for offset in (0, 4):
        assert operations[offset]["line"] < right[offset + 2]["line"] < operations[offset + 1]["line"]
        assert operations[offset + 1]["line"] < right[offset + 3]["line"] < operations[offset + 2]["line"]
    assert not any(r["operation"].startswith("pl.tile.move(")
                   and "pl.Mem.Mat" in ann.get(r["name"], "") for r in cube)
    reference = HERE.parent / "source_after_score_split"
    changed = [p.name for p in reference.glob("*.py")
               if p.read_bytes() != (HERE / "source_candidate_v2" / p.name).read_bytes()]
    assert changed == ["compressor_projection_stages.py"]
    report = {
        "passed": True, "dynamic_axes": 58, "root_tensors": 130, "single_mix": True,
        "actual_allocation_totals": totals, "modified_files": changed,
        "main_mat": mat, "main_left": left, "main_right": right, "main_mma": operations,
        "left_reused_by_two_outputs": dict(usage), "dynamic_compact_acc_preserved": True,
        "main_per_1024_k": {"reference_left_restores": 8, "candidate_left_restores": 4,
                            "reference_mma": 8, "candidate_mma": 8,
                            "reference_mat_loads": 6, "candidate_mat_loads": 6},
        "device_proven": False, "performance_eligible": False,
        "note": "检查资源、操作数及源码顺序，尚不能证明真实流水重叠或完整收益",
    }
    (HERE / f"CODEGEN_B{args.batch}_V2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.batch, "PASS", totals, "Left 8→4 per K1024; Right two slots")


if __name__ == "__main__":
    main()
