# SPDX-License-Identifier: Apache-2.0
"""连续head变换来自已提交0c52ab13；只提取源码变换，不复制hash审计或实验入口。"""

import ast
import copy

BODY = """
tokens = pl.tensor.dim(q_flat, 0)
query_rows = pl.tensor.dim(q_proj_i32, 0)
q_head_input = pl.reshape(q_proj_i32, [query_rows * H, HEAD_DIM])
q_head_output = pl.reshape(q_flat, [tokens * H, HEAD_DIM])
q_head_weights = pl.reshape(wq_b_scale, [H, HEAD_DIM])
q_repeat_rope = pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
q_repeat_scale = pl.tile.full([Q_ROPE_T_TILE, 1], dtype=pl.FP32, value=1.0)
q_flat_i = pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32), pl.FP32)
q_row_i = pl.cast(pl.mul(q_flat_i, 1.0 / ROPE_DIM), pl.INT32, mode="trunc")
q_row_starts = pl.mul(pl.cast(q_row_i, pl.FP32), ROPE_DIM_SCALE)
for dq_work in pl.pipeline(worker_id, tokens * (H // Q_ROPE_T_TILE), WORKERS, stage=2):
    qt = dq_work // (H // Q_ROPE_T_TILE)
    hg = dq_work % (H // Q_ROPE_T_TILE) * Q_ROPE_T_TILE
    head_row = qt * H + hg
    qr_scale_one = pl.load(qr_scale_pad_store, [qt, 0], [1, 1], target_memory=pl.MemorySpace.Vec)
    qr_scale_dq_t = pl.col_expand_mul(q_repeat_scale, qr_scale_one)
    q_cos_one = pl.load(rope_cos_il, [qt, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
    q_sin_one = pl.load(rope_sin_signed, [qt, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
    q_cos_il = pl.col_expand_mul(q_repeat_rope, q_cos_one)
    q_sin_signed = pl.col_expand_mul(q_repeat_rope, q_sin_one)
    q_swap_one = pl.load(rope_swap_idx, [qt, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
    q_swap_repeated = pl.col_expand_mul(q_repeat_rope, pl.cast(q_swap_one, pl.FP32))
    q_abs_swap = pl.cast(pl.add(pl.reshape(q_swap_repeated, [1, Q_ROPE_T_TILE * ROPE_DIM]), q_row_starts), pl.INT32)
    q_flat_gather_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
    q_head_acc = pl.load(q_head_input, [head_row, 0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
    q_head_scale = pl.load(q_head_weights, [hg, 0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
    q_head_dq = pl.mul(q_head_row_scaled, q_head_scale)
    q_head_sq = pl.mul(q_head_dq, q_head_dq)
    q_head_sq_row = pl.row_sum(q_head_sq, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32))
    q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
    q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
    q_head_var = pl.add(q_head_sq_mean, EPS)
    q_head_inv_rms = pl.tile.rsqrt(q_head_var, pl.tile.create([1, Q_ROPE_T_TILE], dtype=pl.FP32))
    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
    q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
    pl.store(q_nope_bf16, [head_row, 0], q_head_output)
    q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
    q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
    q_rope_flat = pl.reshape(q_rope_chunk, [1, Q_ROPE_T_TILE * ROPE_DIM])
    q_swapped_flat = pl.tile.gather(q_rope_flat, q_abs_swap, q_flat_gather_tmp)
    q_rope_swapped = pl.reshape(q_swapped_flat, [Q_ROPE_T_TILE, ROPE_DIM])
    q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
    q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
    pl.store(q_rope_bf16, [head_row, NOPE_DIM], q_head_output)
return q_flat
"""


def transform(source, revision=4):
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    text = ast.get_source_segment(source, original)
    old = text.replace("def q_dequant_rope_stage(", "def _q_dequant_rope_original(", 1)
    new = copy.deepcopy(original)
    new.name = "_q_dequant_rope_contiguous_heads"
    body = BODY
    if revision >= 2:
        for name in ("q_head_input", "q_head_output", "q_head_weights"):
            marker = name + " = pl.reshape("
            assert body.count(marker) == 1
            view_op = "pl.tensor.view" if revision >= 3 else "pl.tensor.reshape"
            body = body.replace(marker, name + " = " + view_op + "(", 1)
    if revision >= 4:
        body = body.replace("q_repeat_scale = pl.tile.full([Q_ROPE_T_TILE, 1], dtype=pl.FP32, value=1.0)\n", "", 1)
        body = body.replace(
            "    qr_scale_one = pl.load(qr_scale_pad_store, [qt, 0], [1, 1], target_memory=pl.MemorySpace.Vec)\n"
            "    qr_scale_dq_t = pl.col_expand_mul(q_repeat_scale, qr_scale_one)\n",
            "    qr_scale_scalar = pl.read(qr_scale_pad_store, [qt, 0])\n",
            1,
        )
        body = body.replace(
            "q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)",
            "q_head_row_scaled = pl.mul(q_head_acc_fp32, qr_scale_scalar)",
            1,
        )
    new.body = ast.parse(body).body
    arguments = ", ".join(n.arg for n in original.args.args)
    wrapper = copy.deepcopy(original)
    wrapper.body = ast.parse(f"""
if pl.tensor.dim(q_flat, 0) % Q_ROPE_T_TILE == 0:
    q_flat = _q_dequant_rope_contiguous_heads({arguments})
else:
    q_flat = _q_dequant_rope_original({arguments})
return q_flat
""").body
    # The original decorator remains immediately before old; the new functions
    # carry their own identical inline(auto_scope=False) decorators.
    result = source.replace(
        text,
        old
        + "\n\n\n"
        + ast.unparse(ast.fix_missing_locations(new))
        + "\n\n\n"
        + ast.unparse(ast.fix_missing_locations(wrapper)),
        1,
    )
    ast.parse(result)
    return result
