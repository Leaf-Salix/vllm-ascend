# SPDX-License-Identifier: Apache-2.0
"""独立候选：一次搬入22段RMS对角线，仍按part0..21左折叠。"""

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
baseline = args.root / "cohorts/eightcard-source-v5/ours"
target = args.root / "candidates/hc-rms-batch-gather-20261102-v3/source" / package
assert not target.exists()
# 所有地址<2**24，FP32构造地址中的整数运算精确；首末有效/填充M8均枚举。
for p16 in (16, 32, 48, 64, 128, 192, 240, 256):
    for t0 in range(0, p16, 8):
        for linear in range(22 * 8):
            part, lane = divmod(linear, 8)
            expected = (part * p16 + t0 + lane) * 16 + t0 % 16 + lane
            actual = int(17.0 * lane + float(t0 * 16 + t0 % 16) + float(part * p16 * 16))
            assert actual == expected and 0 <= actual < 22 * p16 * 16 < 2**24
            assert (part * p16 + t0 + lane) * 16 + int(lane + t0 % 16) == expected
            local_row, local_col = divmod(int(linear * 16.0 + lane + t0 % 16), 16)
            assert local_row == linear and local_col == lane + t0 % 16
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
path = target / "hc_input_rms_stages.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_input_rms_range")
loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "rms_block")
start = next(
    i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "rms_lanes"
)
end = next(
    i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "rms_variance"
)
original = copy.deepcopy(loop.body[start:end])
replacement = ast.parse("""rms_all_lane_i16 = pl.tile.ci(0, [1, RMS_PARTS * T_TILE], dtype=pl.INT16)
rms_all_lane = pl.cast(rms_all_lane_i16, pl.FP32)
rms_all_part = pl.cast(pl.mul(rms_all_lane, 0.125), pl.INT32, mode="trunc")
rms_all_part_f32 = pl.cast(rms_all_part, pl.FP32)
rms_all_row = pl.sub(rms_all_lane, pl.mul(rms_all_part_f32, 8.0))
rms_all_base = pl.add(rms_all_row, pl.cast(pl.cast(rms_t0, pl.INT32), pl.FP32))
rms_all_addr = pl.add(rms_all_base, pl.mul(rms_all_part_f32, pl.cast(pl.cast(t_linear, pl.INT32), pl.FP32)))
rms_all_index = pl.cast(rms_all_addr, pl.INT32, mode="trunc")
rms_all_panels = pl.tile.mgather(sq_partials, rms_all_index, coalesce="row", gather_oob="zero")
rms_all_flat = pl.tile.reshape(rms_all_panels, [1, RMS_PARTS * T_TILE * 16])
rms_all_diagonal_f32 = pl.add(pl.add(pl.mul(rms_all_lane, 16.0), rms_all_row),
    pl.cast(pl.cast(rms_t0 % 16, pl.INT32), pl.FP32))
rms_all_diagonal = pl.cast(rms_all_diagonal_f32, pl.INT32, mode="trunc")
rms_all_scratch = pl.create_tile([1, RMS_PARTS * T_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
rms_all_values = pl.tile.gather(rms_all_flat, rms_all_diagonal, rms_all_scratch)
rms_total = pl.tile.extract(rms_all_values, 0, 0, [1, T_TILE], target_memory=pl.MemorySpace.Vec)
for rms_part in pl.range(1, RMS_PARTS):
    rms_value = pl.tile.extract(rms_all_values, 0, rms_part * T_TILE, [1, T_TILE], target_memory=pl.MemorySpace.Vec)
    rms_total = pl.add(rms_total, rms_value)
""").body
loop.body[start:end] = replacement
inverse = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "native_inverse")
original_one = copy.deepcopy(inverse.value.args[0])
inverse.value.args[0] = ast.parse("pl.tile.full([1, T_TILE], dtype=pl.FP32, value=1.0)", mode="eval").body
original_store = copy.deepcopy(loop.body[-1])
assert isinstance(original_store, ast.Assign) and "inv_rms[" in ast.unparse(original_store.targets[0])
loop.body[-1] = ast.parse("pl.store(pl.reshape(native_inverse, [T_TILE, 1]), [rms_t0, 0], inv_rms)").body[0]
reverted = copy.deepcopy(tree)
f = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
check = next(n for n in ast.walk(f) if isinstance(n, ast.For) and ast.unparse(n.target) == "rms_block")
check.body[start : start + len(replacement)] = original
inverse = next(n for n in check.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "native_inverse")
inverse.value.args[0] = original_one
check.body[-1] = original_store
assert ast.dump(reverted) == ast.dump(before), "RMS sum tree, normalization or stores changed"
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
(target.parents[4] / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
            + " + HC RMS batched diagonal gather",
            "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
            "reverse_ast_equal": True,
            "address_index_exact": True,
            "scope": "row MGather + local diagonal; part0..21 add/sqrt/div identical; not qualified",
        },
        indent=2,
    )
    + "\n"
)
print(target)
