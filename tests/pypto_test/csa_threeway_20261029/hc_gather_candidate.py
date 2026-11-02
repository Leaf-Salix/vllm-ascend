# SPDX-License-Identifier: Apache-2.0
"""单因素候选：HC输入8列重复打包改用Tile gather；不改计算/舍入/信号。"""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
source = args.root / "cohorts/eightcard-source-v5/ours/source" / package
target = args.root / "candidates/hc-gather-20261102-v3/source" / package
assert not target.exists()
shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
p = target / "hc_native_projection_stages.py"
s = p.read_text()
tree = ast.parse(s)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_pack_input_from_bf16")
lines = s.splitlines(keepends=True)
body = "".join(lines[fn.lineno - 1 : fn.end_lineno])
start = body.index("            panel64_transposed_values =")
end = body.index("            pl.store(", start)
replacement = """            # j -> (row, block8, duplicate, lane8)，只重排已有FP32值。
            pack_lanes = pl.tile.arange(0, [1, 8192], dtype=pl.INT16)
            pack_rows = pl.tile.shrs(pack_lanes, 9)
            pack_columns = pl.tile.add(
                pl.tile.mul(pl.tile.shrs(pl.tile.ands(pack_lanes, 511), 4), 8),
                pl.tile.ands(pack_lanes, 7),
            )
            pack_indices16 = pl.tile.add(pl.tile.mul(pack_rows, 256), pack_columns)
            pack_indices_fp32 = pl.tile.cast(pack_indices16, pl.FP32)
            pack_indices = pl.tile.cast(pack_indices_fp32, pl.INT32)
            pack_flat = pl.tile.reshape(panel64_values, [1, 4096])
            pack_scratch = pl.create_tile([1, 8192], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            pack_gathered = pl.tile.gather(pack_flat, pack_indices, pack_scratch)
            panel64_packed_values = pl.tile.reshape(pack_gathered, [16, 512])
"""
body = body[:start] + replacement + body[end:]
lines[fn.lineno - 1 : fn.end_lineno] = [body]
changed = "".join(lines)
check = ast.parse(changed)
# 只允许此函数改变；所有其它函数和导入逐AST一致。
a = [n for n in tree.body if n is not fn]
b = [n for n in check.body if not (isinstance(n, ast.FunctionDef) and n.name == fn.name)]
assert ast.dump(ast.Module(body=a, type_ignores=[])) == ast.dump(ast.Module(body=b, type_ignores=[]))
original = []
for row in range(16):
    for chunk in range(32):
        eight = list(range(row * 256 + chunk * 8, row * 256 + chunk * 8 + 8))
        original.extend(eight)
        original.extend(eight)
candidate = [(j // 512) * 256 + ((j % 512) // 16) * 8 + j % 8 for j in range(8192)]
assert original == candidate and min(candidate) == 0 and max(candidate) == 4095
p.write_text(changed)
manifest = dict(
    csa_source="65bebf7f6b2e4f3f6492fd6463f73106506f3496 + HC input Tile gather only",
    changed_file="hc_native_projection_stages.py",
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    index_map_elements_checked=8192,
)
(target.parents[4] / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(target)
