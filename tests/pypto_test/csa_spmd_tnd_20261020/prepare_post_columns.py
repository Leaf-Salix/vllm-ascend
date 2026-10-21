# SPDX-License-Identifier: Apache-2.0
"""Trial: distribute independent output columns when token rows underfill AIVs.

This changes only the final O_B dequantization and HC post ownership. Every
element keeps its eight-group FP32 fold, BF16 boundary and HC residual order.
The caller still waits for all O_B producers before this stage. All dimensions
are derived from the actual tensor capacity; no fixed query-to-request mapping.
"""

import ast
import difflib
import json
import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/native_wait_candidate_v1"
    destination = here / "runs/post_columns_native_candidate_v2"
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = destination / "output_projection_b_stages.py"
    before = path.read_text()
    name = "o_dequant_hc_post_stage"
    function = next(node for node in ast.parse(before).body if isinstance(node, ast.FunctionDef) and node.name == name)
    lines = before.splitlines(keepends=True)
    start = function.decorator_list[0].lineno - 1
    original_function = "".join(lines[start : function.end_lineno])
    signature = original_function.split("):\n", 1)[0] + "):\n"
    row_function = original_function.replace("def " + name + "(", "def _o_dequant_hc_post_rows(", 1)
    column_signature = signature.replace("def " + name + "(", "def _o_dequant_hc_post_columns(", 1)
    column_signature = column_signature.replace(
        "    WORKERS: pl.constexpr,\n", "    WORKERS: pl.constexpr,\n    COLS: pl.constexpr,\n"
    )
    column_body = '''    """Unique (token, column block) owner; there is no cross-column reduction."""
    tokens = pl.tensor.dim(out_flat, 0)
    blocks_per_row = D // COLS
    for work in pl.range(worker_id, tokens * blocks_per_row, WORKERS):
        b_tb = work // blocks_per_row
        n0 = (work % blocks_per_row) * COLS
        wb_scale = pl.load(wo_b_scale, [n0], [COLS])
        wb_scale_chunk = pl.reshape(wb_scale, [1, COLS])
        acc = pl.tile.full([1, COLS], dtype=pl.FP32, value=0.0)
        for act_g in pl.pipeline(O_GROUPS, stage=2):
            p_col0 = act_g * D + n0
            p_g = pl.load(partials, [b_tb, p_col0], [1, COLS])
            g_scale_scalar = pl.read(act_scale_dq, [act_g, b_tb])
            p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
            acc = pl.add(acc, pl.mul(p_g_f32, g_scale_scalar))
        out_t = pl.col_expand_mul(acc, wb_scale_chunk)
        out_bf16 = pl.create_tensor([1, COLS], dtype=pl.BF16)
        out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
        out_flat = hc_post_row_block(out_bf16, residual_flat, post, comb, out_flat, b_tb, n0, COLS)
    return out_flat
'''
    args = "partials, act_scale_dq, wo_b_scale, residual_flat, post, comb, out_flat, worker_id, WORKERS"
    wrapper = signature + '    """Fill idle AIVs with independent columns; retain row ownership at larger T."""\n'
    wrapper += "    tokens = pl.tensor.dim(out_flat, 0)\n"
    for index, parts in enumerate((8, 4, 2)):
        wrapper += f"    {'if' if index == 0 else 'elif'} tokens * {parts} <= WORKERS:\n"
        wrapper += f"        out_flat = _o_dequant_hc_post_columns({args}, {4096 // parts})\n"
    wrapper += f"    else:\n        out_flat = _o_dequant_hc_post_rows({args})\n    return out_flat\n"
    after = (
        "".join(lines[:start]) + row_function + "\n\n" + column_signature + column_body
        + "\n\n" + wrapper + "".join(lines[function.end_lineno :])
    )
    # Give the generic HC helper its exact column extent. The JIT dependency
    # metadata currently treats create_tensor([1, COLS]) as dynamic across
    # constexpr variants; the explicit same-size view avoids relying on that
    # inference and preserves all input elements and numerical operations.
    marker = "    x_f32 = pl.cast(x_block, pl.FP32)"
    assert after.count(marker) == 1
    after = after.replace(
        marker,
        "    x_f32 = pl.cast(pl.slice(x_block, [1, COLS], [0, 0], valid_shape=[1, COLS], clamp=True), pl.FP32)",
    )
    ast.parse(after)
    path.write_text(after)
    patch = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/" + path.name, tofile="b/" + path.name,
    )
    (here / "post_columns_v2.patch").write_text("".join(patch))
    (here / "post_columns_v2.json").write_text(json.dumps({
        "source": str(destination), "adopted": False,
        "scope": "T*8<=48:512 columns; else T*4<=48:1024; else T*2<=48:2048; otherwise original4096",
        "preserved": (
            "All producer waits, eight-group fold, BF16 intermediate rounding, "
            "HC residual order, output capacity guards"
        ),
        "new_gm_tensors": False,
    }, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
