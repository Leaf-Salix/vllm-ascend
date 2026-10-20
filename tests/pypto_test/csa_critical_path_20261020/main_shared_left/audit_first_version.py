# SPDX-License-Identifier: Apache-2.0
"""保存首版自动K256共享Left但Right退为单槽的证据；不启动设备。"""

import ast
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wave_current_math"))
from audit_codegen import annotated


def main():
    reports = {}
    for batch in (16, 28):
        ir, = (HERE / f"cpu_b{batch}_v1/kernel/passes_dump").glob("*after_AllocateMemoryAddr.py")
        tree = ast.parse(ir.read_text())
        cube = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
        loops = []
        for node in ast.walk(cube):
            if not isinstance(node, ast.For):
                continue
            # 普通Vec占位与窄化Acc各有一次create，不能把同一循环重复计数。
            seed = any(isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                       and st.target.id.startswith("kv_acc_") and isinstance(st.value, ast.Call)
                       and ast.unparse(st.value.func) == "pl.tile.create"
                       and ast.literal_eval(st.value.args[0]) == [64, 64] for st in node.body)
            main = any(isinstance(n, ast.Name) and n.id.startswith("main_wgate") for n in ast.walk(node))
            if seed and main:
                loops.append(node)
        loop, = loops
        records = annotated(loop)
        annotations = {n.target.id: ast.unparse(n.annotation) for n in ast.walk(loop)
                       if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        moves = [r for r in records if r["operation"].startswith("pl.tile.move(")]
        left = [r for r in moves if "pl.Mem.Left" in annotations[r["name"]]]
        right = [r for r in moves if "pl.Mem.Right" in annotations[r["name"]]]
        assert len(left) == 4 and len(right) == 8
        assert {r["memory"][1] for r in right} == {0}
        reports[str(batch)] = {
            "ir": str(ir), "left_move_definitions": left, "right_move_definitions": right,
            "right_slot_count": 1, "right_slot_bytes": 32768, "device_executed": False,
            "note": "静态定义含初始化/累计互斥分支，不能作为实际执行次数；v1未上卡",
        }
    (HERE / "V1_RIGHT_AUDIT.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print("v1两档单Right证据通过")


if __name__ == "__main__":
    main()
