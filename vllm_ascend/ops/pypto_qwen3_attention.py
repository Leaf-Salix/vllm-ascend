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
_BATCH_PAD = 16
_IO_BLOCKS = 5
_IO_BLOCK_WIDTH = 1024
_IO_CHUNK = 256
_QKV_TN = 256
_QKV_TK = 256
_QKV_N_TILE = 512
_QKV_N_SUB = 2
_QKV_K_SPLITS = 5
_QKV_K_SLICE = 1024
_QKV_K_CHUNKS = 4
_Q_N_TILES = 10
_KV_N_TILES = 2
_OUT_N_SPLITS = 10
_OUT_K_SPLITS = 5
_OUT_TN = 512
_OUT_K_SLICE = 1024
_OUT_TK = 64
_OUT_K_CHUNKS = 16
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


@pl.jit.inline(auto_scope=False)
def _qkv_projection_body(
    x: pl.Tensor,
    weight: pl.Tensor,
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
) -> tuple[
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Scalar[pl.TASK_ID],
    pl.Scalar[pl.TASK_ID],
    pl.Scalar[pl.TASK_ID],
]:
    """Project Q/K/V with the tuned PyPTO-Lib decode topology."""
    rows = pl.tensor.dim(x, 0)
    hidden_pad = pl.create_tensor([_BATCH_PAD, _MODEL_HIDDEN], dtype=pl.BF16)
    with pl.spmd(
        _IO_BLOCKS,
        name_hint="qwen3_pad_normalized_input",
        allow_early_resolve=True,
    ) as input_pad_tid:
        block_base = pl.get_block_idx() * _IO_BLOCK_WIDTH
        for chunk in pl.pipeline(_IO_BLOCK_WIDTH // _IO_CHUNK, stage=2):
            col = block_base + chunk * _IO_CHUNK
            tile = pl.fillpad(
                pl.slice(x, [_BATCH_PAD, _IO_CHUNK], [0, col], valid_shape=[rows, _IO_CHUNK]),
                pad_value=pl.PadValue.zero,
            )
            hidden_pad = pl.assemble(hidden_pad, tile, [0, col])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_q_seed", allow_early_resolve=True) as q_seed_tid:
        for tile_index in pl.pipeline(_Q_N_TILES, stage=2):
            col = tile_index * _QKV_N_TILE
            q_proj = pl.assemble(
                q_proj,
                pl.full([_BATCH_PAD, _QKV_N_TILE], dtype=pl.FP32, value=0.0),
                [0, col],
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_kv_seed", allow_early_resolve=True) as kv_seed_tid:
        for tile_index in pl.pipeline(_KV_N_TILES, stage=2):
            col = tile_index * _QKV_N_TILE
            zero = pl.full([_BATCH_PAD, _QKV_N_TILE], dtype=pl.FP32, value=0.0)
            k_proj = pl.assemble(k_proj, zero, [0, col])
            v_proj = pl.assemble(v_proj, zero, [0, col])

    with pl.spmd(
        _Q_N_TILES * _QKV_K_SPLITS,
        name_hint="qwen3_q_proj",
        allow_early_resolve=True,
        deps=[input_pad_tid, q_seed_tid],
    ) as q_proj_tid:
        work = pl.get_block_idx()
        n_region = (work // _QKV_K_SPLITS) * _QKV_N_TILE
        k_base = (work % _QKV_K_SPLITS) * _QKV_K_SLICE
        for n_sub in pl.range(_QKV_N_SUB):
            n0 = n_region + n_sub * _QKV_TN
            acc = pl.matmul(
                hidden_pad[:, k_base : k_base + _QKV_TK],
                weight[n0 : n0 + _QKV_TN, k_base : k_base + _QKV_TK],
                b_trans=True,
                out_dtype=pl.FP32,
            )
            for k_chunk in pl.pipeline(1, _QKV_K_CHUNKS, stage=2):
                k0 = k_base + k_chunk * _QKV_TK
                acc = pl.matmul_acc(
                    acc,
                    hidden_pad[:, k0 : k0 + _QKV_TK],
                    weight[n0 : n0 + _QKV_TN, k0 : k0 + _QKV_TK],
                    b_trans=True,
                )
            q_proj = pl.assemble(q_proj, acc, [0, n0], atomic=pl.AtomicType.Add)

    with pl.spmd(
        _KV_N_TILES * _QKV_K_SPLITS,
        name_hint="qwen3_k_proj",
        allow_early_resolve=True,
        deps=[input_pad_tid, kv_seed_tid],
    ) as k_proj_tid:
        work = pl.get_block_idx()
        n_region = (work // _QKV_K_SPLITS) * _QKV_N_TILE
        k_base = (work % _QKV_K_SPLITS) * _QKV_K_SLICE
        for n_sub in pl.range(_QKV_N_SUB):
            n0 = n_region + n_sub * _QKV_TN
            weight_row = _MODEL_HIDDEN + n0
            acc = pl.matmul(
                hidden_pad[:, k_base : k_base + _QKV_TK],
                weight[weight_row : weight_row + _QKV_TN, k_base : k_base + _QKV_TK],
                b_trans=True,
                out_dtype=pl.FP32,
            )
            for k_chunk in pl.pipeline(1, _QKV_K_CHUNKS, stage=2):
                k0 = k_base + k_chunk * _QKV_TK
                acc = pl.matmul_acc(
                    acc,
                    hidden_pad[:, k0 : k0 + _QKV_TK],
                    weight[weight_row : weight_row + _QKV_TN, k0 : k0 + _QKV_TK],
                    b_trans=True,
                )
            k_proj = pl.assemble(k_proj, acc, [0, n0], atomic=pl.AtomicType.Add)

    with pl.spmd(
        _KV_N_TILES * _QKV_K_SPLITS,
        name_hint="qwen3_v_proj",
        allow_early_resolve=True,
        deps=[input_pad_tid, kv_seed_tid],
    ) as v_proj_tid:
        work = pl.get_block_idx()
        n_region = (work // _QKV_K_SPLITS) * _QKV_N_TILE
        k_base = (work % _QKV_K_SPLITS) * _QKV_K_SLICE
        for n_sub in pl.range(_QKV_N_SUB):
            n0 = n_region + n_sub * _QKV_TN
            weight_row = _MODEL_HIDDEN + _KV_HIDDEN + n0
            acc = pl.matmul(
                hidden_pad[:, k_base : k_base + _QKV_TK],
                weight[weight_row : weight_row + _QKV_TN, k_base : k_base + _QKV_TK],
                b_trans=True,
                out_dtype=pl.FP32,
            )
            for k_chunk in pl.pipeline(1, _QKV_K_CHUNKS, stage=2):
                k0 = k_base + k_chunk * _QKV_TK
                acc = pl.matmul_acc(
                    acc,
                    hidden_pad[:, k0 : k0 + _QKV_TK],
                    weight[weight_row : weight_row + _QKV_TN, k0 : k0 + _QKV_TK],
                    b_trans=True,
                )
            v_proj = pl.assemble(v_proj, acc, [0, n0], atomic=pl.AtomicType.Add)
    return q_proj, k_proj, v_proj, q_proj_tid, k_proj_tid, v_proj_tid


@pl.jit.inline(auto_scope=False)
def _qkv_norm_rope_body(
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    positions: pl.Tensor,
    cos_sin_cache: pl.Tensor,
    slot_mapping: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    query_out: pl.Tensor,
    q_proj_tid: pl.Scalar[pl.TASK_ID],
    k_proj_tid: pl.Scalar[pl.TASK_ID],
    v_proj_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    rows = pl.tensor.dim(positions, 0)

    with pl.spmd(
        rows * _NUM_KV_HEADS,
        name_hint="qwen3_qkv_norm_rope",
        allow_early_resolve=True,
        deps=[q_proj_tid, k_proj_tid, v_proj_tid],
    ) as qkv_finalize_tid:
        work = pl.get_block_idx()
        row = work // _NUM_KV_HEADS
        kv_head = work % _NUM_KV_HEADS
        position = pl.cast(pl.tensor.read(positions, [row]), pl.INDEX)
        cos = pl.cast(pl.slice(cos_sin_cache, [1, _HALF_HEAD_DIM], [position, 0]), target_type=pl.FP32)
        sin = pl.cast(
            pl.slice(cos_sin_cache, [1, _HALF_HEAD_DIM], [position, _HALF_HEAD_DIM]),
            target_type=pl.FP32,
        )
        q_gamma = pl.cast(q_norm_weight, target_type=pl.FP32)
        k_gamma = pl.cast(k_norm_weight, target_type=pl.FP32)

        q_col = kv_head * _Q_PER_KV * _HEAD_DIM
        q_raw = pl.reshape(
            pl.slice(q_proj, [1, _Q_PER_KV * _HEAD_DIM], [row, q_col]),
            [_Q_PER_KV, _HEAD_DIM],
        )
        q_pad = pl.full([_Q_PAD, _HEAD_DIM], dtype=pl.FP32, value=0.0)
        q_pad = pl.assemble(q_pad, q_raw, [0, 0])
        q_sum = pl.row_sum(pl.mul(q_pad, q_pad))
        q_inv = pl.rsqrt(pl.add(pl.mul(q_sum, 1.0 / _HEAD_DIM), _EPS), high_precision=True)
        q_normed = pl.col_expand_mul(pl.row_expand_mul(q_pad, pl.reshape(q_inv, [_Q_PAD, 1])), q_gamma)
        q_first = pl.slice(q_normed, [_Q_PAD, _HALF_HEAD_DIM], [0, 0])
        q_second = pl.slice(q_normed, [_Q_PAD, _HALF_HEAD_DIM], [0, _HALF_HEAD_DIM])
        q_out_row = row * (_NUM_KV_HEADS * _Q_PER_KV) + kv_head * _Q_PER_KV
        query_out = pl.assemble(
            query_out,
            pl.set_validshape(
                pl.cast(pl.sub(pl.col_expand_mul(q_first, cos), pl.col_expand_mul(q_second, sin)), pl.BF16),
                _Q_PER_KV,
                _HALF_HEAD_DIM,
            ),
            [q_out_row, 0],
        )
        query_out = pl.assemble(
            query_out,
            pl.set_validshape(
                pl.cast(pl.add(pl.col_expand_mul(q_second, cos), pl.col_expand_mul(q_first, sin)), pl.BF16),
                _Q_PER_KV,
                _HALF_HEAD_DIM,
            ),
            [q_out_row, _HALF_HEAD_DIM],
        )

        kv_col = kv_head * _HEAD_DIM
        k_value = pl.slice(
            k_proj,
            [_HEAD_TILE_ROWS, _HEAD_DIM],
            [row, kv_col],
            valid_shape=[1, _HEAD_DIM],
        )
        k_sum = pl.row_sum(pl.mul(k_value, k_value))
        k_inv = pl.rsqrt(pl.add(pl.mul(k_sum, 1.0 / _HEAD_DIM), _EPS), high_precision=True)
        k_normed = pl.col_expand_mul(pl.row_expand_mul(k_value, pl.reshape(k_inv, [_HEAD_TILE_ROWS, 1])), k_gamma)
        k_first = pl.slice(k_normed, [_HEAD_TILE_ROWS, _HALF_HEAD_DIM], [0, 0], valid_shape=[1, _HALF_HEAD_DIM])
        k_second = pl.slice(
            k_normed, [_HEAD_TILE_ROWS, _HALF_HEAD_DIM], [0, _HALF_HEAD_DIM], valid_shape=[1, _HALF_HEAD_DIM]
        )
        slot_value = pl.tensor.read(slot_mapping, [row])
        if slot_value >= 0:
            slot = pl.cast(slot_value, pl.INDEX)
            key_cache = pl.assemble(
                key_cache,
                pl.cast(pl.sub(pl.mul(k_first, cos), pl.mul(k_second, sin)), target_type=pl.BF16),
                [slot, kv_col],
            )
            key_cache = pl.assemble(
                key_cache,
                pl.cast(pl.add(pl.mul(k_second, cos), pl.mul(k_first, sin)), target_type=pl.BF16),
                [slot, kv_col + _HALF_HEAD_DIM],
            )
            value_cache = pl.assemble(
                value_cache,
                pl.cast(pl.slice(v_proj, [1, _HEAD_DIM], [row, kv_col]), target_type=pl.BF16),
                [slot, kv_col],
            )
    return query_out, key_cache, value_cache, qkv_finalize_tid


@pl.jit.inline(auto_scope=False)
def _paged_attention_decode_body(
    query_heads: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_table: pl.Tensor,
    seq_lens: pl.Tensor,
    query_start_loc: pl.Tensor,
    attention_pad: pl.Tensor,
    qkv_finalize_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    batch = pl.tensor.dim(seq_lens, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_attention_seed", allow_early_resolve=True) as seed_tid:
        for tile_index in pl.pipeline(_Q_N_TILES, stage=2):
            col = tile_index * _QKV_N_TILE
            attention_pad = pl.assemble(
                attention_pad,
                pl.full([_BATCH_PAD, _QKV_N_TILE], dtype=pl.BF16, value=0.0),
                [0, col],
            )
    with pl.spmd(
        batch * _NUM_KV_HEADS,
        name_hint="qwen3_paged_attention_decode",
        allow_early_resolve=True,
        deps=[qkv_finalize_tid, seed_tid],
    ) as attention_tid:
        work = pl.get_block_idx()
        request = work // _NUM_KV_HEADS
        kv_head = work % _NUM_KV_HEADS
        query_row = pl.cast(pl.tensor.read(query_start_loc, [request]), pl.INDEX)
        context_len = pl.cast(pl.tensor.read(seq_lens, [request]), pl.INDEX)
        context_blocks = (context_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        cache_col = kv_head * _HEAD_DIM
        q_tile = pl.slice(
            query_heads,
            [_Q_PAD, _HEAD_DIM],
            [query_row * (_NUM_KV_HEADS * _Q_PER_KV) + kv_head * _Q_PER_KV, 0],
            valid_shape=[_Q_PER_KV, _HEAD_DIM],
        )
        running_max = pl.full([1, _Q_PAD], dtype=pl.FP32, value=-3.0e38)
        running_sum = pl.full([1, _Q_PAD], dtype=pl.FP32, value=0.0)
        running_out = pl.full([_Q_PAD, _HEAD_DIM], dtype=pl.FP32, value=0.0)
        for block in pl.range(context_blocks):
            physical_block = pl.cast(pl.tensor.read(block_table, [request, block]), pl.INDEX)
            cache_row = physical_block * _BLOCK_SIZE
            key_tile = pl.slice(key_cache, [_BLOCK_SIZE, _HEAD_DIM], [cache_row, cache_col])
            value_tile = pl.slice(value_cache, [_BLOCK_SIZE, _HEAD_DIM], [cache_row, cache_col])
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
            merged_max = pl.maximum(running_max, block_max_nd)
            old_scale = pl.exp(pl.sub(running_max, merged_max))
            new_scale = pl.exp(pl.sub(block_max_nd, merged_max))
            running_sum = pl.add(pl.mul(old_scale, running_sum), pl.mul(new_scale, block_sum_nd))
            running_out = pl.add(
                pl.row_expand_mul(running_out, pl.reshape(old_scale, [_Q_PAD, 1])),
                pl.row_expand_mul(block_out, pl.reshape(new_scale, [_Q_PAD, 1])),
            )
            running_max = merged_max
        normalized = pl.row_expand_div(running_out, pl.reshape(running_sum, [_Q_PAD, 1]))
        context_row = pl.reshape(
            pl.slice(
                pl.cast(normalized, target_type=pl.BF16),
                [_Q_PER_KV, _HEAD_DIM],
                [0, 0],
            ),
            [1, _Q_PER_KV * _HEAD_DIM],
        )
        attention_pad = pl.assemble(
            attention_pad,
            context_row,
            [request, kv_head * _Q_PER_KV * _HEAD_DIM],
        )
    return attention_pad, attention_tid


@pl.jit.inline(auto_scope=False)
def _output_projection_body(
    attention_pad: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    attention_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Tensor:
    rows = pl.tensor.dim(output, 0)
    projected = pl.create_tensor([_BATCH_PAD, _MODEL_HIDDEN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_out_proj_seed", allow_early_resolve=True) as seed_tid:
        for n_split in pl.pipeline(_OUT_N_SPLITS, stage=2):
            n0 = n_split * _OUT_TN
            projected = pl.assemble(
                projected,
                pl.full([_BATCH_PAD, _OUT_TN], dtype=pl.FP32, value=0.0),
                [0, n0],
            )
    with pl.spmd(
        _OUT_N_SPLITS * _OUT_K_SPLITS,
        name_hint="qwen3_out_proj",
        allow_early_resolve=True,
        deps=[attention_tid, seed_tid],
    ) as out_proj_tid:
        work = pl.get_block_idx()
        n0 = (work // _OUT_K_SPLITS) * _OUT_TN
        k_base = (work % _OUT_K_SPLITS) * _OUT_K_SLICE
        acc = pl.matmul(
            attention_pad[:, k_base : k_base + _OUT_TK],
            weight[n0 : n0 + _OUT_TN, k_base : k_base + _OUT_TK],
            b_trans=True,
            out_dtype=pl.FP32,
        )
        for k_chunk in pl.pipeline(1, _OUT_K_CHUNKS, stage=2):
            k0 = k_base + k_chunk * _OUT_TK
            acc = pl.matmul_acc(
                acc,
                attention_pad[:, k0 : k0 + _OUT_TK],
                weight[n0 : n0 + _OUT_TN, k0 : k0 + _OUT_TK],
                b_trans=True,
            )
        projected = pl.assemble(projected, acc, [0, n0], atomic=pl.AtomicType.Add)
    with pl.spmd(_IO_BLOCKS, name_hint="qwen3_attention_output", deps=[out_proj_tid]):
        col = pl.get_block_idx() * _IO_BLOCK_WIDTH
        tile = pl.cast(projected[:, col : col + _IO_BLOCK_WIDTH], target_type=pl.BF16)
        output = pl.assemble(
            output,
            pl.set_validshape(tile, rows, _IO_BLOCK_WIDTH),
            [0, col],
        )
    return output


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
    q_proj = pl.create_tensor([_BATCH_PAD, _MODEL_HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([_BATCH_PAD, _KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([_BATCH_PAD, _KV_HIDDEN], dtype=pl.FP32)
    query_heads = pl.create_tensor([rows * (_NUM_KV_HEADS * _Q_PER_KV), _HEAD_DIM], dtype=pl.BF16)
    attention_pad = pl.create_tensor([_BATCH_PAD, _MODEL_HIDDEN], dtype=pl.BF16)
    with pl.manual_scope():
        q_proj, k_proj, v_proj, q_proj_tid, k_proj_tid, v_proj_tid = _qkv_projection_body(
            normalized_hidden,
            qkv_weight,
            q_proj,
            k_proj,
            v_proj,
        )
        query_heads, key_cache, value_cache, qkv_finalize_tid = _qkv_norm_rope_body(
            q_proj,
            k_proj,
            v_proj,
            q_norm_weight,
            k_norm_weight,
            positions,
            cos_sin_cache,
            slot_mapping,
            key_cache,
            value_cache,
            query_heads,
            q_proj_tid,
            k_proj_tid,
            v_proj_tid,
        )
        attention_pad, attention_tid = _paged_attention_decode_body(
            query_heads,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            query_start_loc,
            attention_pad,
            qkv_finalize_tid,
        )
        output = _output_projection_body(attention_pad, o_proj_weight, output, attention_tid)
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
    if rows != 1:
        raise ValueError("PyPTO Qwen3 decode attention requires one token")
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
