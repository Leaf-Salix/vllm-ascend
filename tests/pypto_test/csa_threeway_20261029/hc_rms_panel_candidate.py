# SPDX-License-Identifier: Apache-2.0
"""HC RMS单panel读取：零拷贝view由顶层传入，数值仍22段顺序相加。"""

import argparse
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
source = args.root / "cohorts/eightcard-source-v5/ours/source" / package
target = args.root / "candidates/hc-rms-panel-20261102-v6/source" / package
assert not target.exists()
shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
p = target / "hc_input_rms_stages.py"
s = p.read_text()
start = s.index("def hc_input_rms_range(")
head, body = s[:start], s[start:]
head += 'RMS_COLS = pl.dynamic("HC_RMS_COLS")\n\n'
# 装饰器需要紧邻原函数；新增常量放在文件原有常量后。
head = head.replace(
    '@pl.jit.inline(auto_scope=False)\nRMS_COLS = pl.dynamic("HC_RMS_COLS")\n\n',
    'RMS_COLS = pl.dynamic("HC_RMS_COLS")\n\n@pl.jit.inline(auto_scope=False)\n',
)
body = body.replace(
    "sq_partials: pl.Tensor[[SQUARE_ROWS, 16], pl.FP32]", "sq_partials: pl.Tensor[[22, RMS_COLS], pl.FP32]"
)
a, b = body.index("        rms_lanes ="), body.index("        rms_variance =")
body = (
    body[:a]
    + """        panel = pl.load(sq_partials, [0, rms_t0 * 16], [22, 128])
        flat = pl.tile.reshape(panel, [1, 22 * 128])
        order = pl.tile.arange(0, [1, 22 * T_TILE], dtype=pl.INT16)
        parts = pl.tile.shrs(order, 3)
        lanes = pl.tile.ands(order, 7)
        offsets16 = pl.add(pl.tile.add(pl.tile.mul(parts, 128), pl.tile.mul(lanes, 17)),
                          pl.cast(rms_t0 % 16, pl.INT16))
        offsets = pl.tile.cast(pl.tile.cast(offsets16, pl.FP32), pl.INT32)
        scratch = pl.create_tile([1, 22 * T_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
        gathered = pl.tile.gather(flat, offsets, scratch)
        values = pl.tile.reshape(gathered, [22, T_TILE])
        rms_total = pl.tile.slice(values, [1, T_TILE], [0, 0])
        for rms_part in pl.range(1, RMS_PARTS):
            rms_value = pl.tile.slice(values, [1, T_TILE], [rms_part, 0])
            rms_total = pl.add(rms_total, rms_value)
"""
    + body[b:]
)
body = body.replace(
    "pl.full([1, T_TILE], dtype=pl.FP32, value=1.0)", "pl.tile.full([1, T_TILE], dtype=pl.FP32, value=1.0)"
)
body = body.replace(
    "inv_rms[rms_t0 : rms_t0 + T_TILE, 0:1] = pl.reshape(native_inverse, [T_TILE, 1])",
    "pl.store(pl.tile.reshape(native_inverse, [T_TILE, 1]), [rms_t0, 0], inv_rms)",
)
result = head + body
ast.parse(result)
p.write_text(result)
v = target / "vector_schedule.py"
s = v.read_text().replace(
    "from .hc_input_rms_stages import SQUARE_ROWS, hc_input_rms_range",
    "from .hc_input_rms_stages import RMS_COLS, hc_input_rms_range",
)
s = s.replace("hc_squares: pl.Tensor[[SQUARE_ROWS, 16], pl.FP32]", "hc_squares: pl.Tensor[[22, RMS_COLS], pl.FP32]")
v.write_text(s)
e = target / "entry.py"
tree = ast.parse(e.read_text())
before = copy.deepcopy(tree)
extra_import = ast.ImportFrom(module="hc_input_rms_stages", names=[ast.alias(name="RMS_COLS")], level=1)
tree.body.insert(1, extra_import)
core = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_csa_core_group")
index = next(i for i, a in enumerate(core.args.args) if ast.unparse(a.annotation) == "pl.constexpr")
core.args.args.insert(
    index, ast.arg(arg="hc_squares_panel", annotation=ast.parse("pl.Tensor[[22, RMS_COLS], pl.FP32]", mode="eval").body)
)
vector_call = next(
    n
    for n in ast.walk(core)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "vector_schedule"
)
vector_index = next(i for i, n in enumerate(vector_call.args) if isinstance(n, ast.Name) and n.id == "hc_squares")
vector_call.args[vector_index] = ast.Name(id="hc_squares_panel", ctx=ast.Load())
roots = [
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("decode_csa_spmd", "decode_csa_spmd_debug")
]
assert len(roots) == 2
for fn in roots:
    i = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "hc_squares"
    )
    fn.body.insert(i + 1, ast.parse("hc_squares_panel = pl.reshape(hc_squares, [22, p16 * 16])").body[0])
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_csa_core_group"
    ]
    assert len(calls) == 1
    for call in calls:
        call.args.insert(index, ast.Name(id="hc_squares_panel", ctx=ast.Load()))
# 完整撤销入口的view/参数传递后，整个entry AST必须保持原样。
reverted = copy.deepcopy(tree)
reverted.body.pop(1)
for fn in reverted.body:
    if not isinstance(fn, ast.FunctionDef):
        continue
    if fn.name == "_csa_core_group":
        fn.args.args.pop(index)
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "vector_schedule"
        )
        call.args[vector_index] = ast.Name(id="hc_squares", ctx=ast.Load())
    if fn.name in ("decode_csa_spmd", "decode_csa_spmd_debug"):
        fn.body = [
            n
            for n in fn.body
            if not (
                isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "hc_squares_panel"
            )
        ]
        for call in [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_csa_core_group"
        ]:
            call.args.pop(index)
assert ast.dump(reverted) == ast.dump(before)
ast.fix_missing_locations(tree)
e.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
for rows in (16, 32, 48, 64, 128, 256):
    for t0 in range(0, rows, 8):
        panel = [part * rows * 16 + t0 * 16 + col for part in range(22) for col in range(128)]
        for part in range(22):
            for lane in range(8):
                assert panel[part * 128 + lane * 17 + t0 % 16] == (part * rows + t0 + lane) * 16 + t0 % 16 + lane
n = dict(
    csa_source="65bebf7f + HC RMS single panel; root GM view; same ordered reduction",
    changed_files={q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in (p, v, e)},
)
(target.parents[4] / "manifest.json").write_text(json.dumps(n, indent=2) + "\n")
print(target)
