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

"""Pure PyPTO L2-kernel implementation of Qwen3 attention.

The operators borrow vLLM-owned tensors. They never retain a weight, allocate
device memory at launch time, or interpret a host pointer. Callable-local
scratch is fixed during prepare, and all graph-visible output storage is
supplied by the OOT adapter so ACLGraph captures stable device addresses.
"""

from functools import lru_cache

import pypto.language as pl
import torch

_ROWS = pl.dynamic("QWEN3_ROWS")
_ROPE_ROWS = pl.dynamic("QWEN3_ROPE_ROWS")
_CACHE_ROWS = pl.dynamic("QWEN3_CACHE_ROWS")
_BATCH = pl.dynamic("QWEN3_BATCH")
_BLOCKS_PER_REQUEST = pl.dynamic("QWEN3_BLOCKS_PER_REQUEST")
_BATCH_PLUS_ONE = pl.dynamic("QWEN3_BATCH_PLUS_ONE")
_HEAD_TILE_ROWS = 8
_MATMUL_ROW_TILE = 16
_MATMUL_COL_TILE = 256
_COL_TILE = 128
_K_TILE = 256
_EPS = 1e-6
_MODEL_HIDDEN = 5120
_KV_HIDDEN = 1024
_QKV_HIDDEN = _MODEL_HIDDEN + 2 * _KV_HIDDEN
_HEAD_DIM = 128
_HALF_HEAD_DIM = 64
_NUM_KV_HEADS = 8
_Q_PER_KV = 5
_Q_PAD = 16
_BLOCK_SIZE = 128
_ATTN_SCALE = 1.0 / (_HEAD_DIM**0.5)


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
                    [row * (_NUM_KV_HEADS * _Q_PER_KV) + head, 0],
                )
                query_out = pl.assemble(
                    query_out,
                    pl.cast(
                        pl.add(pl.mul(second, cos), pl.mul(first, sin)),
                        target_type=pl.BF16,
                    ),
                    [
                        row * (_NUM_KV_HEADS * _Q_PER_KV) + head,
                        _HALF_HEAD_DIM,
                    ],
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


@lru_cache(maxsize=1)
def init() -> None:
    from pypto.torch import init as pypto_init

    pypto_init()


@pl.jit
def _attention_only(
    positions: pl.Tensor[[_ROWS], pl.INT64],
    normalized_hidden: pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[_QKV_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    cos_sin_cache: pl.Tensor[[_ROPE_ROWS, _HEAD_DIM], pl.BF16],
    o_proj_weight: pl.Tensor[[_MODEL_HIDDEN, _MODEL_HIDDEN], pl.BF16],
    slot_mapping: pl.Tensor[[_ROWS], pl.INT32],
    key_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    value_cache: pl.InOut[pl.Tensor[[_CACHE_ROWS, _KV_HIDDEN], pl.BF16]],
    block_table: pl.Tensor[[_BATCH, _BLOCKS_PER_REQUEST], pl.INT32],
    seq_lens: pl.Tensor[[_BATCH], pl.INT32],
    query_start_loc: pl.Tensor[[_BATCH_PLUS_ONE], pl.INT32],
    output: pl.Out[pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]],
) -> pl.Tensor[[_ROWS, _MODEL_HIDDEN], pl.BF16]:
    rows = pl.tensor.dim(normalized_hidden, 0)
    qkv = pl.create_tensor([rows, _QKV_HIDDEN], dtype=pl.BF16)
    qkv = _linear_body(normalized_hidden, qkv_weight, qkv)
    query_heads = pl.create_tensor([rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM], dtype=pl.BF16)
    key = pl.create_tensor([rows, _KV_HIDDEN], dtype=pl.BF16)
    value = pl.create_tensor([rows, _KV_HIDDEN], dtype=pl.BF16)
    query_heads, key, value = _qkv_norm_rope_body(
        qkv, q_norm_weight, k_norm_weight, positions, cos_sin_cache, query_heads, key, value
    )
    key_cache, value_cache = _kv_scatter_contiguous_body(key, value, slot_mapping, key_cache, value_cache)
    attention_heads = pl.create_tensor([rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM], dtype=pl.BF16)
    attention_heads = _paged_attention_body(
        query_heads, key_cache, value_cache, block_table, seq_lens, query_start_loc, attention_heads
    )
    attention_output = pl.reshape(attention_heads, [rows, _MODEL_HIDDEN])
    projected = pl.create_tensor([rows, _MODEL_HIDDEN], dtype=pl.BF16)
    projected = _linear_body(attention_output, o_proj_weight, projected)
    for col_block in pl.spmd(_MODEL_HIDDEN // _MATMUL_COL_TILE, name_hint="qwen3_attention_output"):
        col = col_block * _MATMUL_COL_TILE
        for row in pl.range(0, rows, _MATMUL_ROW_TILE):
            valid_rows = pl.min(_MATMUL_ROW_TILE, rows - row)
            tile = pl.slice(
                projected,
                [_MATMUL_ROW_TILE, _MATMUL_COL_TILE],
                [row, col],
                valid_shape=[valid_rows, _MATMUL_COL_TILE],
            )
            output = pl.assemble(output, tile, [row, col])
    return output


@lru_cache(maxsize=1)
def registered_attention_only_op() -> torch._ops.OpOverload:
    from pypto.torch import register

    return register(_attention_only, "vllm_ascend_pypto::qwen3_attention_only")


def attention_only(
    *,
    positions: torch.Tensor,
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
) -> None:
    rows = normalized_hidden.shape[0]
    if normalized_hidden.shape != (rows, _MODEL_HIDDEN) or normalized_hidden.dtype != torch.bfloat16:
        raise ValueError("PyPTO Qwen3 attention requires BF16 [tokens, 5120] normalized hidden states")
    if positions.shape != (rows,) or positions.dtype != torch.int64:
        raise ValueError("PyPTO Qwen3 attention requires one INT64 position per token")
    if slot_mapping.shape != (rows,) or slot_mapping.dtype != torch.int32:
        raise ValueError("PyPTO Qwen3 attention requires one INT32 cache slot per token")
    expected = (
        (qkv_weight, (_QKV_HIDDEN, _MODEL_HIDDEN), torch.bfloat16),
        (q_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        (k_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        (o_proj_weight, (_MODEL_HIDDEN, _MODEL_HIDDEN), torch.bfloat16),
        (output, (rows, _MODEL_HIDDEN), torch.bfloat16),
    )
    if any(tensor.shape != shape or tensor.dtype != dtype for tensor, shape, dtype in expected):
        raise ValueError("PyPTO Qwen3 attention weight or output shape/dtype mismatch")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != _HEAD_DIM or cos_sin_cache.dtype != torch.bfloat16:
        raise ValueError("PyPTO Qwen3 attention requires a BF16 RoPE cache with width 128")
    if (
        key_cache.ndim != 2
        or key_cache.shape[1] != _KV_HIDDEN
        or key_cache.dtype != torch.bfloat16
        or value_cache.shape != key_cache.shape
        or value_cache.dtype != torch.bfloat16
    ):
        raise ValueError("PyPTO Qwen3 attention requires matching BF16 paged KV caches")
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or seq_lens.shape != (block_table.shape[0],)
        or seq_lens.dtype != torch.int32
        or query_start_loc.shape != (block_table.shape[0] + 1,)
        or query_start_loc.dtype != torch.int32
    ):
        raise ValueError("PyPTO Qwen3 attention requires matching device paging metadata")
    tensors = (
        positions,
        normalized_hidden,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        o_proj_weight,
        slot_mapping,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        output,
    )
    if normalized_hidden.device.type != "npu" or any(
        tensor.device != normalized_hidden.device or not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("PyPTO Qwen3 attention requires contiguous tensors on one device")
    registered_attention_only_op()(
        positions,
        normalized_hidden,
        qkv_weight,
        q_norm_weight.view(1, _HEAD_DIM),
        k_norm_weight.view(1, _HEAD_DIM),
        cos_sin_cache,
        o_proj_weight,
        slot_mapping,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        output,
    )
