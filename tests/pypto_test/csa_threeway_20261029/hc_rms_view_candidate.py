# SPDX-License-Identifier: Apache-2.0
"""单因素：HC RMS显式GM view单panel读取与局部gather，保留22段左折叠及sqrt/div。"""

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
target = args.root / "candidates/hc-rms-view-20261102-v7/source" / package
assert not target.exists()
shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
p = target / "hc_input_rms_stages.py"
s = p.read_text()
tree = ast.parse(s)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_input_rms_range")
lines = s.splitlines(keepends=True)
body = "".join(lines[fn.lineno - 1 : fn.end_lineno])
start = body.index("        rms_lanes =")
end = body.index("        rms_variance =", start)
replacement = """        rms_panel = pl.load(rms_matrix, [0, rms_t0 * 16], [RMS_PARTS, 128])
        rms_flat = pl.tile.reshape(rms_panel, [1, RMS_PARTS * 128])
        rms_order = pl.tile.arange(0, [1, RMS_PARTS * T_TILE], dtype=pl.INT16)
        rms_parts = pl.tile.shrs(rms_order, 3)
        rms_lanes = pl.tile.ands(rms_order, 7)
        rms_offsets16 = pl.add(
            pl.tile.add(pl.tile.mul(rms_parts, 128), pl.tile.mul(rms_lanes, 17)),
            pl.cast(rms_t0 % 16, pl.INT16),
        )
        rms_offsets = pl.tile.cast(pl.tile.cast(rms_offsets16, pl.FP32), pl.INT32)
        rms_scratch = pl.create_tile([1, RMS_PARTS * T_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
        rms_selected = pl.tile.gather(rms_flat, rms_offsets, rms_scratch)
        rms_values = pl.tile.reshape(rms_selected, [RMS_PARTS, T_TILE])
        rms_total = pl.tile.slice(rms_values, [1, T_TILE], [0, 0])
        for rms_part in pl.range(1, RMS_PARTS):
            rms_value = pl.tile.slice(rms_values, [1, T_TILE], [rms_part, 0])
            rms_total = pl.add(rms_total, rms_value)
"""
body = body[:start] + replacement + body[end:]
body = body.replace(
    "    token_tiles =",
    "    rms_matrix = pl.tensor.view(sq_partials, [RMS_PARTS, t_linear * 16])\n    token_tiles =",
)
body = body.replace(
    "pl.full([1, T_TILE], dtype=pl.FP32, value=1.0)", "pl.tile.full([1, T_TILE], dtype=pl.FP32, value=1.0)"
)
body = body.replace(
    "        inv_rms[rms_t0 : rms_t0 + T_TILE, 0:1] = pl.reshape(native_inverse, [T_TILE, 1])",
    "        pl.store(pl.tile.reshape(native_inverse, [T_TILE, 1]), [rms_t0, 0], inv_rms)",
)
lines[fn.lineno - 1 : fn.end_lineno] = [body]
changed = "".join(lines)
check = ast.parse(changed)
a = [n for n in tree.body if n is not fn]
b = [n for n in check.body if not (isinstance(n, ast.FunctionDef) and n.name == fn.name)]
assert ast.dump(ast.Module(body=a, type_ignores=[])) == ast.dump(ast.Module(body=b, type_ignores=[]))
# 独立构造8行块读取的物理地址，检查与旧实现逐token对角线地址一致。
cases = 0
for rows in (16, 32, 48, 64, 128, 256):
    for t0 in range(0, rows, 8):
        panel = [part * rows * 16 + t0 * 16 + col for part in range(22) for col in range(128)]
        for part in range(22):
            for lane in range(8):
                j = part * 8 + lane
                offset = (j >> 3) * 128 + (j & 7) * 17 + t0 % 16
                expected = (part * rows + t0 + lane) * 16 + (t0 % 16 + lane)
                assert panel[offset] == expected
                cases += 1
p.write_text(changed)
manifest = dict(
    csa_source="65bebf7f6b2e4f3f6492fd6463f73106506f3496 + HC RMS tensor.view single panel only",
    changed_file=p.name,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    diagonal_addresses_checked=cases,
)
(target.parents[4] / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(target)
