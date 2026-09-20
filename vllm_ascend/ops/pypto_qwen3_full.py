# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express OR implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Pure PyPTO L2-kernel operators used by experimental Qwen3 full mode.

The operators borrow vLLM-owned tensors.  They never retain a weight, allocate
device memory inside a PyPTO callable, or interpret a host pointer.  All output
storage is supplied explicitly by the OOT adapter so ACLGraph captures stable
device addresses.
"""

from functools import lru_cache

import pypto.language as pl
import torch

_ROWS = pl.dynamic("QWEN3_ROWS")
_INPUT_COLS = pl.dynamic("QWEN3_INPUT_COLS")
_OUTPUT_COLS = pl.dynamic("QWEN3_OUTPUT_COLS")
_HIDDEN_COLS = pl.dynamic("QWEN3_HIDDEN_COLS")
_GATE_COLS = pl.dynamic("QWEN3_GATE_COLS")
_VOCAB_ROWS = pl.dynamic("QWEN3_VOCAB_ROWS")
_ROPE_ROWS = pl.dynamic("QWEN3_ROPE_ROWS")
_CACHE_ROWS = pl.dynamic("QWEN3_CACHE_ROWS")
_VALUE_SPAN = pl.dynamic("QWEN3_VALUE_SPAN")
_BATCH = pl.dynamic("QWEN3_BATCH")
_BLOCKS_PER_REQUEST = pl.dynamic("QWEN3_BLOCKS_PER_REQUEST")
_BATCH_PLUS_ONE = pl.dynamic("QWEN3_BATCH_PLUS_ONE")
_QUERY_HEAD_ROWS = pl.dynamic("QWEN3_QUERY_HEAD_ROWS")

_ROW_TILE = 8
_HEAD_TILE_ROWS = 8
_MATMUL_ROW_TILE = 16
_MATMUL_COL_TILE = 256
_COL_TILE = 128
_ADD_RMS_COL_TILE = 512
_K_TILE = 256
_EPS = 1e-6
_MODEL_HIDDEN = 5120
_MLP_HIDDEN = 17408
_KV_HIDDEN = 1024
_QKV_HIDDEN = _MODEL_HIDDEN + 2 * _KV_HIDDEN
_HEAD_DIM = 128
_HALF_HEAD_DIM = 64
_NUM_KV_HEADS = 8
_Q_PER_KV = 5
_Q_PAD = 16
_PADDED_ATTN_HIDDEN = _NUM_KV_HEADS * _Q_PAD * _HEAD_DIM
_BLOCK_SIZE = 128
_ATTN_SCALE = 1.0 / (_HEAD_DIM**0.5)
_MODES = {"off", "partial", "attention_block", "full"}


def mode() -> str:
    from vllm_ascend import envs

    value = envs.VLLM_ASCEND_PYPTO_QWEN3_MODE
    if value not in _MODES:
        raise ValueError(f"VLLM_ASCEND_PYPTO_QWEN3_MODE must be one of {sorted(_MODES)}, got {value!r}")
    return value


def enabled() -> bool:
    return mode() == "full"


@pl.jit.inline
def _linear_body(x: pl.Tensor, weight: pl.Tensor, out: pl.Tensor) -> pl.Tensor:
    # Every supported Qwen3 projection has K aligned to _K_TILE.  Keeping that
    # contract explicit avoids dynamic tail handling in every reduction tile.
    rows = pl.tensor.dim(x, 0)
    input_cols = pl.tensor.dim(x, 1)
    output_cols = pl.tensor.dim(weight, 0)
    for block in pl.spmd(pl.system.available_cluster_count(), name_hint="qwen3_linear"):
        block_count = pl.tile.get_block_num()
        for row in pl.range(0, rows, _MATMUL_ROW_TILE):
            valid_rows = pl.min(_MATMUL_ROW_TILE, rows - row)
            for col in pl.range(
                block * _MATMUL_COL_TILE,
                output_cols,
                block_count * _MATMUL_COL_TILE,
            ):
                valid_cols = pl.min(_MATMUL_COL_TILE, output_cols - col)
                x_first = pl.slice(
                    x,
                    [_MATMUL_ROW_TILE, _K_TILE],
                    [row, 0],
                    valid_shape=[valid_rows, _K_TILE],
                )
                w_first = pl.slice(
                    weight,
                    [_MATMUL_COL_TILE, _K_TILE],
                    [col, 0],
                    valid_shape=[valid_cols, _K_TILE],
                )
                acc = pl.matmul(x_first, w_first, b_trans=True, out_dtype=pl.FP32)
                for k0 in pl.range(_K_TILE, input_cols, _K_TILE):
                    x_tile = pl.slice(
                        x,
                        [_MATMUL_ROW_TILE, _K_TILE],
                        [row, k0],
                        valid_shape=[valid_rows, _K_TILE],
                    )
                    w_tile = pl.slice(
                        weight,
                        [_MATMUL_COL_TILE, _K_TILE],
                        [col, k0],
                        valid_shape=[valid_cols, _K_TILE],
                    )
                    acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True)
                out = pl.assemble(out, pl.cast(acc, target_type=pl.BF16), [row, col])
    return out


@pl.jit
def _linear(
    x: pl.Tensor[[_ROWS, _INPUT_COLS], pl.BF16],
    weight: pl.Tensor[[_OUTPUT_COLS, _INPUT_COLS], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _OUTPUT_COLS], pl.BF16]],
) -> pl.Tensor[[_ROWS, _OUTPUT_COLS], pl.BF16]:
    out = _linear_body(x, weight, out)
    return out


@pl.jit.inline
def _qkv_norm_rope_body(
    qkv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    positions: pl.Tensor,
    cos_sin_cache: pl.Tensor,
    query_out: pl.Tensor,
    key_out: pl.Tensor,
    value_out: pl.Tensor,
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    rows = pl.tensor.dim(qkv, 0)

    for row in pl.parallel(rows):
        position = pl.cast(pl.tensor.read(positions, [row]), pl.INDEX)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_qkv_norm_rope"):
            cos = pl.cast(
                pl.slice(cos_sin_cache, [1, _HALF_HEAD_DIM], [position, 0]),
                target_type=pl.FP32,
            )
            sin = pl.cast(
                pl.slice(
                    cos_sin_cache,
                    [1, _HALF_HEAD_DIM],
                    [position, _HALF_HEAD_DIM],
                ),
                target_type=pl.FP32,
            )
            q_gamma = pl.cast(q_norm_weight, target_type=pl.FP32)
            k_gamma = pl.cast(k_norm_weight, target_type=pl.FP32)

            for head in pl.range(_MODEL_HIDDEN // _HEAD_DIM):
                offset = head * _HEAD_DIM
                head_value = pl.cast(
                    pl.slice(
                        qkv,
                        [_HEAD_TILE_ROWS, _HEAD_DIM],
                        [row, offset],
                        valid_shape=[1, _HEAD_DIM],
                    ),
                    target_type=pl.FP32,
                )
                square_sum = pl.row_sum(pl.mul(head_value, head_value))
                inv_rms = pl.rsqrt(
                    pl.add(pl.mul(square_sum, 1.0 / _HEAD_DIM), _EPS),
                    high_precision=True,
                )
                normalized = pl.col_expand_mul(
                    pl.row_expand_mul(
                        head_value,
                        pl.reshape(inv_rms, [_HEAD_TILE_ROWS, 1]),
                    ),
                    q_gamma,
                )
                first = pl.slice(
                    normalized,
                    [_HEAD_TILE_ROWS, _HALF_HEAD_DIM],
                    [0, 0],
                    valid_shape=[1, _HALF_HEAD_DIM],
                )
                second = pl.slice(
                    normalized,
                    [_HEAD_TILE_ROWS, _HALF_HEAD_DIM],
                    [0, _HALF_HEAD_DIM],
                    valid_shape=[1, _HALF_HEAD_DIM],
                )
                query_out = pl.assemble(
                    query_out,
                    pl.cast(
                        pl.sub(pl.mul(first, cos), pl.mul(second, sin)),
                        target_type=pl.BF16,
                    ),
                    [row, offset],
                )
                query_out = pl.assemble(
                    query_out,
                    pl.cast(
                        pl.add(pl.mul(second, cos), pl.mul(first, sin)),
                        target_type=pl.BF16,
                    ),
                    [row, offset + _HALF_HEAD_DIM],
                )

            for head in pl.range(_KV_HIDDEN // _HEAD_DIM):
                qkv_offset = _MODEL_HIDDEN + head * _HEAD_DIM
                output_offset = head * _HEAD_DIM
                head_value = pl.cast(
                    pl.slice(
                        qkv,
                        [_HEAD_TILE_ROWS, _HEAD_DIM],
                        [row, qkv_offset],
                        valid_shape=[1, _HEAD_DIM],
                    ),
                    target_type=pl.FP32,
                )
                square_sum = pl.row_sum(pl.mul(head_value, head_value))
                inv_rms = pl.rsqrt(
                    pl.add(pl.mul(square_sum, 1.0 / _HEAD_DIM), _EPS),
                    high_precision=True,
                )
                normalized = pl.col_expand_mul(
                    pl.row_expand_mul(
                        head_value,
                        pl.reshape(inv_rms, [_HEAD_TILE_ROWS, 1]),
                    ),
                    k_gamma,
                )
                first = pl.slice(
                    normalized,
                    [_HEAD_TILE_ROWS, _HALF_HEAD_DIM],
                    [0, 0],
                    valid_shape=[1, _HALF_HEAD_DIM],
                )
                second = pl.slice(
                    normalized,
                    [_HEAD_TILE_ROWS, _HALF_HEAD_DIM],
                    [0, _HALF_HEAD_DIM],
                    valid_shape=[1, _HALF_HEAD_DIM],
                )
                key_out = pl.assemble(
                    key_out,
                    pl.cast(
                        pl.sub(pl.mul(first, cos), pl.mul(second, sin)),
                        target_type=pl.BF16,
                    ),
                    [row, output_offset],
                )
                key_out = pl.assemble(
                    key_out,
                    pl.cast(
                        pl.add(pl.mul(second, cos), pl.mul(first, sin)),
                        target_type=pl.BF16,
                    ),
                    [row, output_offset + _HALF_HEAD_DIM],
                )

            value_out = pl.assemble(
                value_out,
                pl.slice(
                    qkv,
                    [1, _KV_HIDDEN],
                    [row, _MODEL_HIDDEN + _KV_HIDDEN],
                ),
                [row, 0],
            )
    return query_out, key_out, value_out


@pl.jit
def _qkv_norm_rope(
    hidden_states: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[_QKV_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    positions: pl.Tensor[[_ROWS], pl.INT64],
    cos_sin_cache: pl.Tensor[[_ROPE_ROWS, _HEAD_DIM], pl.BF16],
    query_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
    key_out: pl.Out[pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16]],
    value_out: pl.Out[pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16]],
) -> tuple[
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16],
    pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16],
]:
    rows = pl.tensor.dim(hidden_states, 0)
    qkv = pl.create_tensor([rows, _QKV_HIDDEN], dtype=pl.BF16)
    qkv = _linear_body(hidden_states, qkv_weight, qkv)
    query_out, key_out, value_out = _qkv_norm_rope_body(
        qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        query_out,
        key_out,
        value_out,
    )
    return query_out, key_out, value_out


@pl.jit.inline
def _rms_norm_body(x: pl.Tensor, weight: pl.Tensor, out: pl.Tensor) -> pl.Tensor:
    rows = pl.tensor.dim(x, 0)
    for row in pl.parallel(0, rows, _ROW_TILE):
        valid_rows = pl.min(_ROW_TILE, rows - row)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_rms_norm"):
            sq_sum = pl.full([1, _ROW_TILE], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, _MODEL_HIDDEN, _COL_TILE):
                x_tile = pl.cast(
                    pl.slice(
                        x,
                        [_ROW_TILE, _COL_TILE],
                        [row, col],
                        valid_shape=[valid_rows, _COL_TILE],
                    ),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_tile, x_tile)), [1, _ROW_TILE]))
            variance = pl.reshape(pl.mul(sq_sum, 1.0 / _MODEL_HIDDEN), [_ROW_TILE, 1])
            inv_rms = pl.rsqrt(pl.add(variance, _EPS), high_precision=True)
            for col in pl.range(0, _MODEL_HIDDEN, _COL_TILE):
                x_tile = pl.cast(
                    pl.slice(
                        x,
                        [_ROW_TILE, _COL_TILE],
                        [row, col],
                        valid_shape=[valid_rows, _COL_TILE],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.cast(
                    pl.slice(weight, [1, _COL_TILE], [0, col]),
                    target_type=pl.FP32,
                )
                normalized = pl.col_expand_mul(pl.row_expand_mul(x_tile, inv_rms), gamma)
                out = pl.assemble(out, pl.cast(normalized, target_type=pl.BF16), [row, col])
    return out


@pl.jit
def _rms_norm(
    x: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, _MODEL_HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]:
    out = _rms_norm_body(x, weight, out)
    return out


@pl.jit.inline
def _add_rms_norm_body(
    x: pl.Tensor,
    residual: pl.Tensor,
    weight: pl.Tensor,
    norm_out: pl.Tensor,
    residual_out: pl.Tensor,
) -> tuple[pl.Tensor, pl.Tensor]:
    rows = pl.tensor.dim(x, 0)
    for row in pl.parallel(0, rows, _ROW_TILE):
        valid_rows = pl.min(_ROW_TILE, rows - row)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_add_rms_norm"):
            sq_sum = pl.full([1, _ROW_TILE], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, _MODEL_HIDDEN, _ADD_RMS_COL_TILE):
                x_tile = pl.cast(
                    pl.slice(
                        x, [_ROW_TILE, _ADD_RMS_COL_TILE], [row, col], valid_shape=[valid_rows, _ADD_RMS_COL_TILE]
                    ),
                    target_type=pl.FP32,
                )
                residual_tile = pl.cast(
                    pl.slice(
                        residual,
                        [_ROW_TILE, _ADD_RMS_COL_TILE],
                        [row, col],
                        valid_shape=[valid_rows, _ADD_RMS_COL_TILE],
                    ),
                    target_type=pl.FP32,
                )
                added = pl.add(x_tile, residual_tile)
                # torch_npu.npu_add_rms_norm publishes a BF16 residual using
                # round-to-nearest-even. Its normalized output is also closest
                # to consuming that narrowed value; keep both outputs on the
                # same intermediate rather than normalizing a different sum.
                added_bf16 = pl.cast(added, target_type=pl.BF16, mode="rint")
                added_for_norm = pl.cast(added_bf16, target_type=pl.FP32)
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(added_for_norm, added_for_norm)), [1, _ROW_TILE]),
                )
                residual_out = pl.assemble(
                    residual_out,
                    added_bf16,
                    [row, col],
                )
            variance = pl.reshape(pl.mul(sq_sum, 1.0 / _MODEL_HIDDEN), [_ROW_TILE, 1])
            inv_rms = pl.rsqrt(pl.add(variance, _EPS), high_precision=True)
            for col in pl.range(0, _MODEL_HIDDEN, _ADD_RMS_COL_TILE):
                x_tile = pl.cast(
                    pl.slice(
                        x,
                        [_ROW_TILE, _ADD_RMS_COL_TILE],
                        [row, col],
                        valid_shape=[valid_rows, _ADD_RMS_COL_TILE],
                    ),
                    target_type=pl.FP32,
                )
                residual_tile = pl.cast(
                    pl.slice(
                        residual,
                        [_ROW_TILE, _ADD_RMS_COL_TILE],
                        [row, col],
                        valid_shape=[valid_rows, _ADD_RMS_COL_TILE],
                    ),
                    target_type=pl.FP32,
                )
                added = pl.cast(
                    pl.cast(pl.add(x_tile, residual_tile), target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
                gamma = pl.cast(
                    pl.slice(weight, [1, _ADD_RMS_COL_TILE], [0, col]),
                    target_type=pl.FP32,
                )
                normalized = pl.col_expand_mul(pl.row_expand_mul(added, inv_rms), gamma)
                norm_out = pl.assemble(norm_out, pl.cast(normalized, target_type=pl.BF16), [row, col])
    return norm_out, residual_out


@pl.jit
def _add_rms_norm(
    x: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    residual: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, _MODEL_HIDDEN], pl.BF16],
    norm_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
    residual_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> tuple[
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
]:
    norm_out, residual_out = _add_rms_norm_body(x, residual, weight, norm_out, residual_out)
    return norm_out, residual_out


@pl.jit.inline
def _silu_and_mul_body(x: pl.Tensor, out: pl.Tensor) -> pl.Tensor:
    rows = pl.tensor.dim(x, 0)
    output_cols = pl.tensor.dim(out, 1)
    for row in pl.parallel(0, rows, _ROW_TILE):
        valid_rows = pl.min(_ROW_TILE, rows - row)
        for col in pl.range(0, output_cols, _COL_TILE):
            valid_cols = pl.min(_COL_TILE, output_cols - col)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_silu_and_mul"):
                gate = pl.cast(
                    pl.slice(x, [_ROW_TILE, _COL_TILE], [row, col], valid_shape=[valid_rows, valid_cols]),
                    target_type=pl.FP32,
                )
                up = pl.cast(
                    pl.slice(
                        x,
                        [_ROW_TILE, _COL_TILE],
                        [row, output_cols + col],
                        valid_shape=[valid_rows, valid_cols],
                    ),
                    target_type=pl.FP32,
                )
                silu = pl.mul(gate, pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0)))
                out = pl.assemble(out, pl.cast(pl.mul(silu, up), target_type=pl.BF16), [row, col])
    return out


@pl.jit
def _silu_and_mul(
    x: pl.Tensor[[_ROWS, _GATE_COLS], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _OUTPUT_COLS], pl.BF16]],
) -> pl.Tensor[[_ROWS, _OUTPUT_COLS], pl.BF16]:
    out = _silu_and_mul_body(x, out)
    return out


@pl.jit
def _mlp(
    x: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    gate_up_weight: pl.Tensor[[2 * _MLP_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    down_weight: pl.Tensor[[_MODEL_HIDDEN, _MLP_HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]:
    rows = pl.tensor.dim(x, 0)
    gate_up = pl.create_tensor([rows, 2 * _MLP_HIDDEN], dtype=pl.BF16)
    activated = pl.create_tensor([rows, _MLP_HIDDEN], dtype=pl.BF16)
    gate_up = _linear_body(x, gate_up_weight, gate_up)
    activated = _silu_and_mul_body(gate_up, activated)
    out = _linear_body(activated, down_weight, out)
    return out


@pl.jit
def _embedding(
    token_ids: pl.Tensor[[_ROWS], pl.INT32],
    weight: pl.Tensor[[_VOCAB_ROWS, _HIDDEN_COLS], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _HIDDEN_COLS], pl.BF16]],
) -> pl.Tensor[[_ROWS, _HIDDEN_COLS], pl.BF16]:
    rows = pl.tensor.dim(token_ids, 0)
    hidden = pl.tensor.dim(weight, 1)
    for row in pl.parallel(rows):
        token = pl.cast(pl.tensor.read(token_ids, [row]), pl.INDEX)
        for col in pl.range(0, hidden, _COL_TILE):
            valid_cols = pl.min(_COL_TILE, hidden - col)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_embedding"):
                value = pl.slice(weight, [1, _COL_TILE], [token, col], valid_shape=[1, valid_cols])
                out = pl.assemble(out, value, [row, col])
    return out


@pl.jit
def _rope(
    positions: pl.Tensor[[_ROWS], pl.INT64],
    query: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    key: pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16],
    cos_sin_cache: pl.Tensor[[_ROPE_ROWS, _HEAD_DIM], pl.BF16],
    query_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
    key_out: pl.Out[pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16]],
) -> tuple[
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16],
]:
    rows = pl.tensor.dim(query, 0)
    for row in pl.parallel(rows):
        position = pl.cast(pl.tensor.read(positions, [row]), pl.INDEX)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_rope"):
            cos = pl.cast(pl.slice(cos_sin_cache, [1, _HALF_HEAD_DIM], [position, 0]), pl.FP32)
            sin = pl.cast(
                pl.slice(cos_sin_cache, [1, _HALF_HEAD_DIM], [position, _HALF_HEAD_DIM]),
                pl.FP32,
            )
            for head in pl.range(_MODEL_HIDDEN // _HEAD_DIM):
                offset = head * _HEAD_DIM
                first = pl.cast(pl.slice(query, [1, _HALF_HEAD_DIM], [row, offset]), pl.FP32)
                second = pl.cast(
                    pl.slice(query, [1, _HALF_HEAD_DIM], [row, offset + _HALF_HEAD_DIM]),
                    pl.FP32,
                )
                query_out = pl.assemble(
                    query_out,
                    pl.cast(pl.sub(pl.mul(first, cos), pl.mul(second, sin)), pl.BF16),
                    [row, offset],
                )
                query_out = pl.assemble(
                    query_out,
                    pl.cast(pl.add(pl.mul(second, cos), pl.mul(first, sin)), pl.BF16),
                    [row, offset + _HALF_HEAD_DIM],
                )
            for head in pl.range(_KV_HIDDEN // _HEAD_DIM):
                offset = head * _HEAD_DIM
                first = pl.cast(pl.slice(key, [1, _HALF_HEAD_DIM], [row, offset]), pl.FP32)
                second = pl.cast(
                    pl.slice(key, [1, _HALF_HEAD_DIM], [row, offset + _HALF_HEAD_DIM]),
                    pl.FP32,
                )
                key_out = pl.assemble(
                    key_out,
                    pl.cast(pl.sub(pl.mul(first, cos), pl.mul(second, sin)), pl.BF16),
                    [row, offset],
                )
                key_out = pl.assemble(
                    key_out,
                    pl.cast(pl.add(pl.mul(second, cos), pl.mul(first, sin)), pl.BF16),
                    [row, offset + _HALF_HEAD_DIM],
                )
    return query_out, key_out


@pl.jit.inline
def _kv_scatter_body(
    key: pl.Tensor,
    value_storage_span: pl.Tensor,
    value_row_stride: pl.Scalar[pl.INT32],
    slot_mapping: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
) -> tuple[pl.Tensor, pl.Tensor]:
    rows = pl.tensor.dim(key, 0)
    for row in pl.parallel(rows):
        slot = pl.cast(pl.tensor.read(slot_mapping, [row]), pl.INDEX)
        for col in pl.range(0, _KV_HIDDEN, _COL_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_kv_scatter"):
                key_tile = pl.slice(key, [1, _COL_TILE], [row, col])
                value_tile = pl.slice(
                    value_storage_span,
                    [_COL_TILE],
                    [row * value_row_stride + col],
                )
                value_tile = pl.reshape(value_tile, [1, _COL_TILE])
                key_cache = pl.assemble(key_cache, key_tile, [slot, col])
                value_cache = pl.assemble(value_cache, value_tile, [slot, col])
    return key_cache, value_cache


@pl.jit.inline
def _kv_scatter_contiguous_body(
    key: pl.Tensor,
    value: pl.Tensor,
    slot_mapping: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
) -> tuple[pl.Tensor, pl.Tensor]:
    rows = pl.tensor.dim(key, 0)
    for row in pl.parallel(rows):
        slot_value = pl.tensor.read(slot_mapping, [row])
        if slot_value >= 0:
            slot = pl.cast(slot_value, pl.INDEX)
            for col in pl.range(0, _KV_HIDDEN, _COL_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_kv_scatter_contiguous"):
                    key_cache = pl.assemble(
                        key_cache,
                        pl.slice(key, [1, _COL_TILE], [row, col]),
                        [slot, col],
                    )
                    value_cache = pl.assemble(
                        value_cache,
                        pl.slice(value, [1, _COL_TILE], [row, col]),
                        [slot, col],
                    )
    return key_cache, value_cache


@pl.jit
def _kv_scatter(
    key: pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16],
    value_storage_span: pl.Tensor[[_VALUE_SPAN], pl.BF16],
    value_row_stride: pl.Scalar[pl.INT32],
    slot_mapping: pl.Tensor[[_ROWS], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
) -> tuple[
    pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16],
    pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16],
]:
    key_cache, value_cache = _kv_scatter_body(
        key,
        value_storage_span,
        value_row_stride,
        slot_mapping,
        key_cache,
        value_cache,
    )
    return key_cache, value_cache


@pl.jit
def _attention_pad(
    query: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    padded: pl.Out[pl.Tensor[[_ROWS, _PADDED_ATTN_HIDDEN], pl.BF16]],
) -> pl.Tensor[[_ROWS, _PADDED_ATTN_HIDDEN], pl.BF16]:
    rows = pl.tensor.dim(query, 0)
    for row in pl.parallel(rows):
        for kv_head in pl.range(_NUM_KV_HEADS):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_attention_pad"):
                for query_head in pl.range(_Q_PER_KV):
                    value = pl.slice(
                        query,
                        [1, _HEAD_DIM],
                        [row, (kv_head * _Q_PER_KV + query_head) * _HEAD_DIM],
                    )
                    padded = pl.assemble(
                        padded,
                        value,
                        [row, (kv_head * _Q_PAD + query_head) * _HEAD_DIM],
                    )
    return padded


@pl.jit.inline
def _query_heads_body(
    query: pl.Tensor,
    query_heads: pl.Tensor,
) -> pl.Tensor:
    rows = pl.tensor.dim(query, 0)
    for row in pl.parallel(rows):
        for query_head in pl.range(_NUM_KV_HEADS * _Q_PER_KV):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_query_heads"):
                query_heads = pl.assemble(
                    query_heads,
                    pl.slice(
                        query,
                        [1, _HEAD_DIM],
                        [row, query_head * _HEAD_DIM],
                    ),
                    [row * (_NUM_KV_HEADS * _Q_PER_KV) + query_head, 0],
                )
    return query_heads


@pl.jit.inline
def _paged_attention_body(
    query_heads: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_table: pl.Tensor,
    seq_lens: pl.Tensor,
    query_start_loc: pl.Tensor,
    heads_out: pl.Tensor,
) -> pl.Tensor:
    batch = pl.tensor.dim(seq_lens, 0)
    for request in pl.range(batch):
        query_start = pl.cast(pl.tensor.read(query_start_loc, [request]), pl.INDEX)
        query_end = pl.cast(pl.tensor.read(query_start_loc, [request + 1]), pl.INDEX)
        query_len = query_end - query_start
        sequence_len = pl.cast(pl.tensor.read(seq_lens, [request]), pl.INDEX)
        prefix_len = sequence_len - query_len
        for query_offset in pl.range(query_len):
            query_row = query_start + query_offset
            context_len = prefix_len + query_offset + 1
            context_blocks = (context_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            for kv_head in pl.range(_NUM_KV_HEADS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_paged_attention"):
                    q_tile = pl.slice(
                        query_heads,
                        [_Q_PAD, _HEAD_DIM],
                        [query_row * (_NUM_KV_HEADS * _Q_PER_KV) + kv_head * _Q_PER_KV, 0],
                        valid_shape=[_Q_PER_KV, _HEAD_DIM],
                    )
                    first_physical_block = pl.cast(pl.tensor.read(block_table, [request, 0]), pl.INDEX)
                    first_cache_row = first_physical_block * _BLOCK_SIZE
                    cache_col = kv_head * _HEAD_DIM
                    first_key_tile = pl.slice(
                        key_cache,
                        [_BLOCK_SIZE, _HEAD_DIM],
                        [first_cache_row, cache_col],
                    )
                    first_value_tile = pl.slice(
                        value_cache,
                        [_BLOCK_SIZE, _HEAD_DIM],
                        [first_cache_row, cache_col],
                    )
                    first_valid_len = pl.min(_BLOCK_SIZE, context_len)
                    first_scores = pl.matmul(q_tile, first_key_tile, b_trans=True, out_dtype=pl.FP32)
                    first_scores = pl.fillpad(
                        pl.set_validshape(
                            pl.mul(first_scores, _ATTN_SCALE),
                            _Q_PER_KV,
                            first_valid_len,
                        ),
                        pad_value=pl.PadValue.min,
                    )
                    first_max = pl.row_max(first_scores)
                    first_probabilities = pl.exp(pl.row_expand_sub(first_scores, first_max))
                    first_probabilities_bf16 = pl.cast(first_probabilities, target_type=pl.BF16)
                    first_sum = pl.row_sum(pl.cast(first_probabilities_bf16, target_type=pl.FP32))
                    first_out = pl.matmul(first_probabilities_bf16, first_value_tile, out_dtype=pl.FP32)
                    first_scale_nd = pl.full([1, _Q_PAD], dtype=pl.FP32, value=1.0)
                    running_out = pl.row_expand_mul(first_out, pl.reshape(first_scale_nd, [_Q_PAD, 1]))
                    running_max = pl.reshape(first_max, [1, _Q_PAD])
                    running_sum = pl.reshape(first_sum, [1, _Q_PAD])
                    for block in pl.range(1, context_blocks):
                        physical_block = pl.cast(pl.tensor.read(block_table, [request, block]), pl.INDEX)
                        cache_row = physical_block * _BLOCK_SIZE
                        key_tile = pl.slice(
                            key_cache,
                            [_BLOCK_SIZE, _HEAD_DIM],
                            [cache_row, cache_col],
                        )
                        value_tile = pl.slice(
                            value_cache,
                            [_BLOCK_SIZE, _HEAD_DIM],
                            [cache_row, cache_col],
                        )
                        valid_len = pl.min(_BLOCK_SIZE, context_len - block * _BLOCK_SIZE)
                        scores = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.FP32)
                        scores = pl.fillpad(
                            pl.set_validshape(pl.mul(scores, _ATTN_SCALE), _Q_PER_KV, valid_len),
                            pad_value=pl.PadValue.min,
                        )
                        block_max = pl.row_max(scores)
                        probabilities = pl.exp(pl.row_expand_sub(scores, block_max))
                        probabilities_bf16 = pl.cast(probabilities, target_type=pl.BF16)
                        block_sum = pl.row_sum(pl.cast(probabilities_bf16, target_type=pl.FP32))
                        block_out = pl.matmul(probabilities_bf16, value_tile, out_dtype=pl.FP32)
                        block_max_nd = pl.reshape(block_max, [1, _Q_PAD])
                        block_sum_nd = pl.reshape(block_sum, [1, _Q_PAD])
                        merged_max_nd = pl.maximum(running_max, block_max_nd)
                        old_scale_nd = pl.exp(pl.sub(running_max, merged_max_nd))
                        new_scale_nd = pl.exp(pl.sub(block_max_nd, merged_max_nd))
                        running_sum = pl.add(
                            pl.mul(old_scale_nd, running_sum),
                            pl.mul(new_scale_nd, block_sum_nd),
                        )
                        old_scale = pl.reshape(old_scale_nd, [_Q_PAD, 1])
                        new_scale = pl.reshape(new_scale_nd, [_Q_PAD, 1])
                        running_out = pl.add(
                            pl.row_expand_mul(running_out, old_scale),
                            pl.row_expand_mul(block_out, new_scale),
                        )
                        running_max = merged_max_nd
                    normalized = pl.row_expand_div(running_out, pl.reshape(running_sum, [_Q_PAD, 1]))
                    heads_out = pl.assemble(
                        heads_out,
                        pl.set_validshape(
                            pl.cast(normalized, pl.BF16),
                            _Q_PER_KV,
                            _HEAD_DIM,
                        ),
                        [query_row * (_NUM_KV_HEADS * _Q_PER_KV) + kv_head * _Q_PER_KV, 0],
                    )
    return heads_out


@pl.jit
def _paged_attention(
    query_heads: pl.Tensor[[_QUERY_HEAD_ROWS, _HEAD_DIM], pl.BF16],
    key_cache: pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16],
    value_cache: pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16],
    block_table: pl.Tensor[[_BATCH, _BLOCKS_PER_REQUEST], pl.INT32],
    seq_lens: pl.Tensor[[_BATCH], pl.INT32],
    query_start_loc: pl.Tensor[[_BATCH_PLUS_ONE], pl.INT32],
    heads_out: pl.Out[pl.Tensor[[_QUERY_HEAD_ROWS, _HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[_QUERY_HEAD_ROWS, _HEAD_DIM], pl.BF16]:
    heads_out = _paged_attention_body(
        query_heads,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        heads_out,
    )
    return heads_out


@pl.jit
def _kv_scatter_paged_attention(
    query_heads: pl.Tensor[[_QUERY_HEAD_ROWS, _HEAD_DIM], pl.BF16],
    key: pl.Tensor[[_ROWS, _KV_HIDDEN], pl.BF16],
    value_storage_span: pl.Tensor[[_VALUE_SPAN], pl.BF16],
    value_row_stride: pl.Scalar[pl.INT32],
    slot_mapping: pl.Tensor[[_ROWS], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    block_table: pl.Tensor[[_BATCH, _BLOCKS_PER_REQUEST], pl.INT32],
    seq_lens: pl.Tensor[[_BATCH], pl.INT32],
    query_start_loc: pl.Tensor[[_BATCH_PLUS_ONE], pl.INT32],
    heads_out: pl.Out[pl.Tensor[[_QUERY_HEAD_ROWS, _HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[_QUERY_HEAD_ROWS, _HEAD_DIM], pl.BF16]:
    key_cache, value_cache = _kv_scatter_body(
        key,
        value_storage_span,
        value_row_stride,
        slot_mapping,
        key_cache,
        value_cache,
    )
    heads_out = _paged_attention_body(
        query_heads,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        heads_out,
    )
    return heads_out


@pl.jit
def _attention_residual_block_first(
    positions: pl.Tensor[[_ROWS], pl.INT64],
    hidden_states: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    input_norm_weight: pl.Tensor[[1, _MODEL_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[_QKV_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    cos_sin_cache: pl.Tensor[[_ROPE_ROWS, _HEAD_DIM], pl.BF16],
    o_proj_weight: pl.Tensor[[_MODEL_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    post_attention_norm_weight: pl.Tensor[[1, _MODEL_HIDDEN], pl.BF16],
    slot_mapping: pl.Tensor[[_ROWS], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    block_table: pl.Tensor[[_BATCH, _BLOCKS_PER_REQUEST], pl.INT32],
    seq_lens: pl.Tensor[[_BATCH], pl.INT32],
    query_start_loc: pl.Tensor[[_BATCH_PLUS_ONE], pl.INT32],
    mlp_input_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
    updated_residual_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> tuple[
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
]:
    rows = pl.tensor.dim(hidden_states, 0)
    normalized_hidden_states = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    normalized_hidden_states = _rms_norm_body(
        hidden_states,
        input_norm_weight,
        normalized_hidden_states,
    )
    query = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    key = pl.create_tensor([rows, _KV_HIDDEN], dtype=pl.BF16)
    value = pl.create_tensor([rows, _KV_HIDDEN], dtype=pl.BF16)
    qkv = pl.create_tensor([rows, _QKV_HIDDEN], dtype=pl.BF16)
    qkv = _linear_body(normalized_hidden_states, qkv_weight, qkv)
    query, key, value = _qkv_norm_rope_body(
        qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        query,
        key,
        value,
    )
    key_cache, value_cache = _kv_scatter_contiguous_body(
        key,
        value,
        slot_mapping,
        key_cache,
        value_cache,
    )
    query_heads = pl.create_tensor(
        [rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM],
        dtype=pl.BF16,
    )
    query_heads = _query_heads_body(query, query_heads)
    attention_heads = pl.create_tensor(
        [rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM],
        dtype=pl.BF16,
    )
    attention_heads = _paged_attention_body(
        query_heads,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        attention_heads,
    )
    attention_output = pl.reshape(attention_heads, [rows, _MODEL_HIDDEN])
    projected = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    projected = _linear_body(attention_output, o_proj_weight, projected)
    mlp_input_out, updated_residual_out = _add_rms_norm_body(
        projected,
        hidden_states,
        post_attention_norm_weight,
        mlp_input_out,
        updated_residual_out,
    )
    return mlp_input_out, updated_residual_out


@pl.jit
def _attention_residual_block(
    positions: pl.Tensor[[_ROWS], pl.INT64],
    hidden_states: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    residual: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    input_norm_weight: pl.Tensor[[1, _MODEL_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[_QKV_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    cos_sin_cache: pl.Tensor[[_ROPE_ROWS, _HEAD_DIM], pl.BF16],
    o_proj_weight: pl.Tensor[[_MODEL_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    post_attention_norm_weight: pl.Tensor[[1, _MODEL_HIDDEN], pl.BF16],
    slot_mapping: pl.Tensor[[_ROWS], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    block_table: pl.Tensor[[_BATCH, _BLOCKS_PER_REQUEST], pl.INT32],
    seq_lens: pl.Tensor[[_BATCH], pl.INT32],
    query_start_loc: pl.Tensor[[_BATCH_PLUS_ONE], pl.INT32],
    mlp_input_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
    updated_residual_out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> tuple[
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
]:
    rows = pl.tensor.dim(hidden_states, 0)
    normalized_hidden_states = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    attention_residual = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    normalized_hidden_states, attention_residual = _add_rms_norm_body(
        hidden_states,
        residual,
        input_norm_weight,
        normalized_hidden_states,
        attention_residual,
    )
    query = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    key = pl.create_tensor([rows, _KV_HIDDEN], dtype=pl.BF16)
    value = pl.create_tensor([rows, _KV_HIDDEN], dtype=pl.BF16)
    qkv = pl.create_tensor([rows, _QKV_HIDDEN], dtype=pl.BF16)
    qkv = _linear_body(normalized_hidden_states, qkv_weight, qkv)
    query, key, value = _qkv_norm_rope_body(
        qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        query,
        key,
        value,
    )
    key_cache, value_cache = _kv_scatter_contiguous_body(
        key,
        value,
        slot_mapping,
        key_cache,
        value_cache,
    )
    query_heads = pl.create_tensor(
        [rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM],
        dtype=pl.BF16,
    )
    query_heads = _query_heads_body(query, query_heads)
    attention_heads = pl.create_tensor(
        [rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM],
        dtype=pl.BF16,
    )
    attention_heads = _paged_attention_body(
        query_heads,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        attention_heads,
    )
    attention_output = pl.reshape(attention_heads, [rows, _MODEL_HIDDEN])
    projected = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    projected = _linear_body(attention_output, o_proj_weight, projected)
    mlp_input_out, updated_residual_out = _add_rms_norm_body(
        projected,
        attention_residual,
        post_attention_norm_weight,
        mlp_input_out,
        updated_residual_out,
    )
    return mlp_input_out, updated_residual_out


@pl.jit
def _attention_unpad(
    padded: pl.Tensor[[_ROWS, _PADDED_ATTN_HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]:
    rows = pl.tensor.dim(padded, 0)
    for row in pl.parallel(rows):
        for kv_head in pl.range(_NUM_KV_HEADS):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_attention_unpad"):
                for query_head in pl.range(_Q_PER_KV):
                    value = pl.slice(
                        padded,
                        [1, _HEAD_DIM],
                        [row, (kv_head * _Q_PAD + query_head) * _HEAD_DIM],
                    )
                    out = pl.assemble(
                        out,
                        value,
                        [row, (kv_head * _Q_PER_KV + query_head) * _HEAD_DIM],
                    )
    return out


@lru_cache(maxsize=1)
def registered_ops() -> dict[str, torch._ops.OpOverload]:
    from pypto.torch import register

    return {
        "linear": register(_linear, "vllm_ascend_pypto::qwen3_linear"),
        "qkv_norm_rope": register(
            _qkv_norm_rope,
            "vllm_ascend_pypto::qwen3_qkv_norm_rope",
        ),
        "rms_norm": register(_rms_norm, "vllm_ascend_pypto::qwen3_rms_norm"),
        "add_rms_norm": register(_add_rms_norm, "vllm_ascend_pypto::qwen3_add_rms_norm"),
        "silu_and_mul": register(_silu_and_mul, "vllm_ascend_pypto::qwen3_silu_and_mul"),
        "mlp": register(_mlp, "vllm_ascend_pypto::qwen3_mlp"),
        "embedding": register(_embedding, "vllm_ascend_pypto::qwen3_embedding"),
        "rope": register(_rope, "vllm_ascend_pypto::qwen3_rope"),
        "kv_scatter": register(_kv_scatter, "vllm_ascend_pypto::qwen3_kv_scatter"),
        "attention_pad": register(_attention_pad, "vllm_ascend_pypto::qwen3_attention_pad"),
        "paged_attention": register(_paged_attention, "vllm_ascend_pypto::qwen3_paged_attention"),
        "kv_scatter_paged_attention": register(
            _kv_scatter_paged_attention,
            "vllm_ascend_pypto::qwen3_kv_scatter_paged_attention",
        ),
        "attention_unpad": register(_attention_unpad, "vllm_ascend_pypto::qwen3_attention_unpad"),
    }


@lru_cache(maxsize=1)
def registered_attention_block_ops() -> dict[str, torch._ops.OpOverload]:
    from pypto.torch import register

    return {
        "first": register(
            _attention_residual_block_first,
            "vllm_ascend_pypto::qwen3_attention_residual_block_first",
        ),
        "regular": register(
            _attention_residual_block,
            "vllm_ascend_pypto::qwen3_attention_residual_block",
        ),
    }


def init() -> None:
    from pypto.torch import init as pypto_init

    from vllm_ascend.ops.pypto_swimlane import init_pypto

    init_pypto(pypto_init)


def attention_residual_block(
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    input_norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    mlp_input_out: torch.Tensor | None = None,
    updated_residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("PyPTO Qwen3 attention block requires 2D BF16 hidden states")
    rows = hidden_states.shape[0]
    if hidden_states.shape[1] != _MODEL_HIDDEN:
        raise ValueError("PyPTO Qwen3 attention block requires hidden size 5120")
    if residual is not None and (residual.shape != hidden_states.shape or residual.dtype != torch.bfloat16):
        raise ValueError("PyPTO Qwen3 attention block residual must match hidden states")
    expected_shapes = (
        (input_norm_weight, (_MODEL_HIDDEN,), "input RMSNorm weight"),
        (qkv_weight, (_QKV_HIDDEN, _MODEL_HIDDEN), "QKV weight"),
        (q_norm_weight, (_HEAD_DIM,), "query RMSNorm weight"),
        (k_norm_weight, (_HEAD_DIM,), "key RMSNorm weight"),
        (o_proj_weight, (_MODEL_HIDDEN, _MODEL_HIDDEN), "output projection weight"),
        (post_attention_norm_weight, (_MODEL_HIDDEN,), "post-attention RMSNorm weight"),
    )
    for tensor, shape, name in expected_shapes:
        if tensor.dtype != torch.bfloat16 or tensor.shape != shape:
            raise ValueError(f"PyPTO Qwen3 attention block {name} must be BF16 {shape}")
    if positions.dtype != torch.int64 or positions.shape != (rows,):
        raise ValueError("PyPTO Qwen3 attention block requires one INT64 position per row")
    if cos_sin_cache.dtype != torch.bfloat16 or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != _HEAD_DIM:
        raise ValueError("PyPTO Qwen3 attention block requires a BF16 [positions, 128] RoPE cache")
    if slot_mapping.dtype != torch.int32 or slot_mapping.shape != (rows,):
        raise ValueError("PyPTO Qwen3 attention block requires one INT32 cache slot per row")
    if (
        key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or key_cache.ndim != 2
        or key_cache.shape[1] != _KV_HIDDEN
        or value_cache.shape != key_cache.shape
    ):
        raise ValueError("PyPTO Qwen3 attention block requires matching BF16 [cache rows, 1024] K/V caches")
    batch = seq_lens.shape[0] if seq_lens.ndim == 1 else -1
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or seq_lens.dtype != torch.int32
        or query_start_loc.dtype != torch.int32
        or query_start_loc.shape != (batch + 1,)
    ):
        raise ValueError("PyPTO Qwen3 attention block paging metadata shape or dtype mismatch")
    tensors = (
        positions,
        hidden_states,
        input_norm_weight,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        o_proj_weight,
        post_attention_norm_weight,
        slot_mapping,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
    )
    if residual is not None:
        tensors += (residual,)
    if hidden_states.device.type != "npu" or any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("PyPTO Qwen3 attention block requires all tensors on one NPU device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("PyPTO Qwen3 attention block requires contiguous tensors")

    mlp_input = torch.empty_like(hidden_states) if mlp_input_out is None else mlp_input_out
    updated_residual = torch.empty_like(hidden_states) if updated_residual_out is None else updated_residual_out
    if mlp_input.shape != hidden_states.shape or updated_residual.shape != hidden_states.shape:
        raise ValueError("PyPTO Qwen3 attention block output buffers must match hidden states")
    if mlp_input.dtype != torch.bfloat16 or updated_residual.dtype != torch.bfloat16:
        raise ValueError("PyPTO Qwen3 attention block output buffers must be BF16")
    if mlp_input.device != hidden_states.device or updated_residual.device != hidden_states.device:
        raise ValueError("PyPTO Qwen3 attention block output buffers must be on the input device")
    if not mlp_input.is_contiguous() or not updated_residual.is_contiguous():
        raise ValueError("PyPTO Qwen3 attention block output buffers must be contiguous")
    common_args = (
        positions,
        hidden_states,
        input_norm_weight.view(1, _MODEL_HIDDEN),
        qkv_weight,
        q_norm_weight.view(1, _HEAD_DIM),
        k_norm_weight.view(1, _HEAD_DIM),
        cos_sin_cache,
        o_proj_weight,
        post_attention_norm_weight.view(1, _MODEL_HIDDEN),
        slot_mapping,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        mlp_input,
        updated_residual,
    )
    ops = registered_attention_block_ops()
    if residual is None:
        return ops["first"](*common_args)
    return ops["regular"](
        positions,
        hidden_states,
        residual,
        *common_args[2:],
    )


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.ndim != 2 or weight.ndim != 2:
        raise ValueError("PyPTO Qwen3 linear requires 2D BF16 input and weight")
    if x.shape[1] != weight.shape[1] or x.shape[1] % _K_TILE or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("PyPTO Qwen3 linear requires contiguous [M,K] input and [N,K] weight with K % 256 == 0")
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    return registered_ops()["linear"](x, weight, out)


def qkv_norm_rope(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires 2D BF16 hidden states")
    if hidden_states.shape[1] != _MODEL_HIDDEN:
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires hidden size 5120")
    rows = hidden_states.shape[0]
    if qkv_weight.dtype != torch.bfloat16 or qkv_weight.shape != (_QKV_HIDDEN, _MODEL_HIDDEN):
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires BF16 [7168, 5120] weight")
    if q_norm_weight.dtype != torch.bfloat16 or q_norm_weight.shape != (_HEAD_DIM,):
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires BF16 [128] query RMS weight")
    if k_norm_weight.dtype != torch.bfloat16 or k_norm_weight.shape != (_HEAD_DIM,):
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires BF16 [128] key RMS weight")
    if positions.dtype != torch.int64 or positions.shape != (rows,):
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires one INT64 position per row")
    if cos_sin_cache.dtype != torch.bfloat16 or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != _HEAD_DIM:
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires BF16 [positions, 128] RoPE cache")
    tensors = (
        hidden_states,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
    )
    if hidden_states.device.type != "npu" or any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires tensors on one NPU device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("PyPTO Qwen3 fused QKV front-end requires contiguous tensors")
    query = torch.empty((rows, _MODEL_HIDDEN), dtype=torch.bfloat16, device=hidden_states.device)
    key = torch.empty((rows, _KV_HIDDEN), dtype=torch.bfloat16, device=hidden_states.device)
    value = torch.empty_like(key)
    return registered_ops()["qkv_norm_rope"](
        hidden_states,
        qkv_weight,
        q_norm_weight.view(1, _HEAD_DIM),
        k_norm_weight.view(1, _HEAD_DIM),
        positions,
        cos_sin_cache,
        query,
        key,
        value,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("PyPTO Qwen3 RMSNorm requires 2D BF16 input and BF16 weight")
    if weight.shape != (x.shape[1],) or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("PyPTO Qwen3 RMSNorm input/weight layout mismatch")
    out = torch.empty_like(x)
    return registered_ops()["rms_norm"](x, weight.view(1, -1), out)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != residual.shape or x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("PyPTO Qwen3 add-RMSNorm requires matching BF16 input and residual")
    if x.ndim != 2 or weight.dtype != torch.bfloat16 or weight.shape != (x.shape[1],):
        raise ValueError("PyPTO Qwen3 add-RMSNorm input/weight shape mismatch")
    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("PyPTO Qwen3 add-RMSNorm requires contiguous tensors")
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    return registered_ops()["add_rms_norm"](x, residual, weight.view(1, -1), norm_out, residual_out)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 2 or not x.is_contiguous():
        raise ValueError("PyPTO Qwen3 SwiGLU requires contiguous 2D BF16 input with an even last dimension")
    out = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
    return registered_ops()["silu_and_mul"](x, out)


def mlp(x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != _MODEL_HIDDEN:
        raise ValueError("PyPTO Qwen3 fused MLP requires BF16 [tokens, 5120] input")
    if gate_up_weight.dtype != torch.bfloat16 or gate_up_weight.shape != (2 * _MLP_HIDDEN, _MODEL_HIDDEN):
        raise ValueError("PyPTO Qwen3 fused MLP requires BF16 [34816, 5120] gate/up weight")
    if down_weight.dtype != torch.bfloat16 or down_weight.shape != (_MODEL_HIDDEN, _MLP_HIDDEN):
        raise ValueError("PyPTO Qwen3 fused MLP requires BF16 [5120, 17408] down weight")
    tensors = (x, gate_up_weight, down_weight)
    if x.device.type != "npu" or any(tensor.device != x.device for tensor in tensors):
        raise ValueError("PyPTO Qwen3 fused MLP requires tensors on one NPU device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("PyPTO Qwen3 fused MLP requires contiguous tensors")
    out = torch.empty_like(x)
    return registered_ops()["mlp"](x, gate_up_weight, down_weight, out)


def embedding(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if token_ids.dtype != torch.int32 or token_ids.ndim != 1 or weight.dtype != torch.bfloat16:
        raise ValueError("PyPTO Qwen3 embedding requires 1D INT32 ids and BF16 weight")
    if weight.ndim != 2 or not token_ids.is_contiguous() or not weight.is_contiguous():
        raise ValueError("PyPTO Qwen3 embedding requires contiguous tensors")
    out = torch.empty((token_ids.shape[0], weight.shape[1]), dtype=weight.dtype, device=weight.device)
    return registered_ops()["embedding"](token_ids, weight, out)
