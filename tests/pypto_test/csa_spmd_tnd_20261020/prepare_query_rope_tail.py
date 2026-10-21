# SPDX-License-Identifier: Apache-2.0
"""Try a masked M8 Indexer RoPE tail using the existing full-tile arithmetic."""

import ast
import difflib
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from prepare_score_owner_harness import replace_once
from prepare_short_half_ub import function


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/borrowed_combined_candidate_v106"
    target = here / "runs/query_rope_tail_candidate_v118"
    original = (source / "indexer_query_stages.py").read_text()
    name = "indexer_query_dequant_rope_stage"
    node = function(original, name)
    branch = next(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.If) and ast.unparse(item.test) == "dq_t0 + DEQUANT_T_TILE <= bs"
    )
    lines = original.splitlines(keepends=True)
    block = "".join(lines[branch.body[0].lineno - 1 : branch.body[-1].end_lineno])
    for tensor, width in (("qr_scale", "1"), ("cos", "ROPE_HEAD_DIM"), ("sin", "ROPE_HEAD_DIM")):
        old = f"pl.load({tensor}, [dq_t0, 0], [DEQUANT_T_TILE, {width}])"
        new = f"pl.load({tensor}, [dq_t0, 0], [DEQUANT_T_TILE, {width}], valid_shape=[valid_tail_rows, {width}])"
        block = replace_once(block, old, new)
    block = replace_once(
        block,
        "pl.load(qr_acc_pad, [dq_t0, h0], [DEQUANT_T_TILE, IDX_HEAD_DIM]),",
        "pl.load(qr_acc_pad, [dq_t0, h0], [DEQUANT_T_TILE, IDX_HEAD_DIM], "
        "valid_shape=[valid_tail_rows, IDX_HEAD_DIM]),",
    )
    # All rows are independent. Widen only UB descriptors after masked loads;
    # discarded padding rows never contribute to any valid row or GM store.
    block = replace_once(
        block,
        "                flat_i = ",
        "                qr_scale_tile = pl.set_validshape(qr_scale_tile, DEQUANT_T_TILE, 1)\n"
        "                cos_tile = pl.set_validshape(cos_tile, DEQUANT_T_TILE, ROPE_HEAD_DIM)\n"
        "                sin_tile = pl.set_validshape(sin_tile, DEQUANT_T_TILE, ROPE_HEAD_DIM)\n"
        "                flat_i = ",
    )
    block = replace_once(
        block,
        "                    qr_dequant = ",
        "                    acc_fp32 = pl.set_validshape(acc_fp32, DEQUANT_T_TILE, IDX_HEAD_DIM)\n"
        "                    qr_dequant = ",
    )
    block = replace_once(
        block,
        "                    pl.store(qr_nope_bf16,",
        "                    qr_nope_bf16 = pl.set_validshape(qr_nope_bf16, valid_tail_rows, IDX_NOPE_HEAD_DIM)\n"
        "                    rope_bf16 = pl.set_validshape(rope_bf16, valid_tail_rows, ROPE_HEAD_DIM)\n"
        "                    pl.store(qr_nope_bf16,",
    )
    assigned = {
        item.id
        for item in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }
    for variable in sorted(assigned, key=len, reverse=True):
        block = re.sub(r"\b" + re.escape(variable) + r"\b", "tail_tile_" + variable, block)
    lines[branch.orelse[0].lineno - 1 : branch.orelse[-1].end_lineno] = [
        "                valid_tail_rows = bs - dq_t0\n",
        block,
    ]
    revised = "".join(lines).replace(
        "                # 不读越过真实T的完整8行；不足8行沿用原逐行数学，合法行逐个写回。\n",
        "                # GM只读取实际行；复用完整M8逐元素数学，最后仅写有效行。\n",
    )
    after = function(revised, name)
    updated = next(
        item
        for item in ast.walk(after)
        if isinstance(item, ast.If) and ast.unparse(item.test) == ast.unparse(branch.test)
    )
    assert ast.dump(ast.Module(body=branch.body, type_ignores=[])) == ast.dump(
        ast.Module(body=updated.body, type_ignores=[])
    )
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "indexer_query_stages.py").write_text(revised)
    workspace = next(parent for parent in here.parents if (parent / "env-dsv4-0251rc1.sh").is_file())
    subprocess.run([str(workspace / ".venv/bin/ruff"), "format", str(target / "indexer_query_stages.py")], check=True)
    revised = (target / "indexer_query_stages.py").read_text()
    (here / "runs/query_rope_tail_v118.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                revised.splitlines(True),
                fromfile="a/indexer_query_stages.py",
                tofile="b/indexer_query_stages.py",
            )
        )
    )
    report = {
        "source": str(target),
        "base": str(source),
        "adopted": False,
        "device_execution": False,
        "changed_files": ["indexer_query_stages.py"],
        "full_tile_math_ast_equal": True,
        "masking": "Four input loads narrowed to bs-dq_t0; both BF16 stores narrowed to the same valid rows",
        "arithmetic": "Same FP32 multiply/multiply and RoPE multiply/multiply/add, followed by BF16 rint",
        "source_manifest": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    (here / "query_rope_tail_candidate_v118.json").write_text(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
