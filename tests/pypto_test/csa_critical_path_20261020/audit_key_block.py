# SPDX-License-Identifier: Apache-2.0
"""核对当前集中准备Key候选的符号IR，设备验证须另行执行。"""
import argparse
import ast
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("kernel", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
p = next((args.kernel / "passes_dump").glob("*AllocateMemoryAddr*"))
tree = ast.parse(p.read_text())
function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "cube_schedule")
loads, math_lines, types = [], [], []
for node in ast.walk(function):
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id.startswith("buf_key_l1_pool"):
            types.append(ast.unparse(node.annotation))
        if node.target.id.startswith(("buf_scores_inline", "buf_reduced_pair", "buf_last_reduced")):
            if isinstance(node.value, ast.Call) and "matmul" in ast.unparse(node.value.func):
                math_lines.append(node.lineno)
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.gather_row":
        if ast.unparse(node.args[0]).startswith("buf_key_l1_pool"):
            loads.append({"line": node.lineno, "row": ast.literal_eval(node.args[2])[0]})
assert len(loads) == 64  # 每页两个64B对齐分支；一次实际只执行其中一个。
assert sorted({item["row"] for item in loads}) == list(range(0, 1024, 32))
assert max(item["line"] for item in loads) < min(math_lines)
refs = set(re.findall(r"MemRef\(([^,]+), pl.const\((\d+), pl.INT64\), (\d+)\)", "\n".join(types)))
assert len(refs) == 1 and next(iter(refs))[2] == "131072", refs
result = {"passed": True, "source": str(p), "memory_aliases": [list(ref) for ref in refs],
          "load_sites": len(loads), "runtime_pages_per_block": 32,
          "all_key_loads_before_qk": True, "device_validated": False}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
