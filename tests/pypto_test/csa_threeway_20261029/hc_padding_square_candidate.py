# SPDX-License-Identifier: Apache-2.0
"""独立候选：只跳过真实请求末端之后的HC平方块，保留有效M16计算树。"""

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
target = args.root / "candidates/hc-padding-square-20261102-v1/source" / package
assert not target.exists()
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
path = target / "hc_input_rms_stages.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_input_square_range")
fn.args.args.append(ast.arg(arg="live_end", annotation=ast.parse("pl.Scalar[pl.INDEX]", mode="eval").body))
loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "sq_group_item")
original_body = copy.deepcopy(loop.body)
assert ast.unparse(loop.body[0].targets[0]) == "sq_part"
branch = ast.parse("""if sq_t0 < live_end:
    pass
else:
    sq_out_row = sq_part * t_linear + sq_t0
    sq_partials[sq_out_row:sq_out_row + 16, 0:16] = pl.full([16, 16], dtype=pl.FP32, value=0.0)
""").body[0]
branch.body = loop.body[1:]
loop.body = [loop.body[0], branch]
reverted = copy.deepcopy(tree)
f = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
f.args.args.pop()
check_loop = next(n for n in ast.walk(f) if isinstance(n, ast.For) and ast.unparse(n.target) == "sq_group_item")
check_loop.body = original_body
assert ast.dump(reverted) == ast.dump(before), "active square computation changed"
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")

path = target / "cube_schedule.py"
text = path.read_text()
line = "    hc_squares = hc_input_square_range(x_fp32, hc_squares, worker)\n"
assert text.count(line) == 1
scan = """    # 只省掉最后有效query之后的整M16块；中间dummy仍保守计算。
    square_live_end = tokens
    if tokens >= 128:
        square_live_end = pl.cast(0, pl.INDEX)
        square_metadata_valid = pl.cast(1, pl.INT32)
        for square_request in pl.range(pl.tensor.dim(idx_lengths, 0)):
            square_begin = pl.cast(pl.read(idx_bounds, [square_request]), pl.INDEX)
            square_end = pl.cast(pl.read(idx_bounds, [square_request + 1]), pl.INDEX)
            square_length = pl.read(idx_lengths, [square_request])
            if square_begin < 0 or square_end < square_begin or square_end > tokens or square_length < 0:
                square_metadata_valid = pl.cast(0, pl.INT32)
            if square_length > 0:
                square_live_end = pl.max(square_live_end, square_end)
        if square_metadata_valid == 0:
            square_live_end = tokens
    hc_squares = hc_input_square_range(x_fp32, hc_squares, worker, square_live_end)
"""
path.write_text(text.replace(line, scan))
assert path.read_text().replace(scan, line) == text
(target.parents[4] / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
            + " + skip trailing dummy HC square blocks",
            "changed_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, target / "hc_input_rms_stages.py")
            },
            "reverse_ast_equal": True,
            "same_activity_domains_only": True,
            "hardware_blocked_reason": "Cube split changed MIX members; internal routing and completion not qualified",
            "scope": "diagnostic fixture only; require all five Q/L domains equal; not qualified",
        },
        indent=2,
    )
    + "\n"
)
print(target)
