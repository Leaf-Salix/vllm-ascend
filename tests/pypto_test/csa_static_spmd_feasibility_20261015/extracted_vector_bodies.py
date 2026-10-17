# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""静态 CSA 的计算体拆分准备：真实 RoPE 符号和 Native KV 回写，未接生产入口。"""

import argparse
import json
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from pypto.runtime import RunConfig

TOKENS = pl.dynamic("STATIC_BODY_TOKENS")
CACHE_ROWS = pl.dynamic("STATIC_BODY_CACHE_ROWS")
ROPE_DIM = 64
HEAD_DIM = 512
BLOCK_SIZE = 32
ROPE_ROWS = 4
WRITE_ROWS = 8


@pl.jit.inline(auto_scope=False)
def rope_sign_body(
    freqs_sin: pl.Tensor[[TOKENS, ROPE_DIM], pl.FP32],
    idx_sin_signed: pl.Out[pl.Tensor[[TOKENS, ROPE_DIM], pl.FP32]],
    logical_worker: pl.Scalar[pl.INDEX],
    logical_workers: pl.Scalar[pl.INDEX],
):
    """来自冻结 baseline decode_csa.py:351；运算和尾行处理保持不变。

    唯一调度变换是把 get_block_idx() 和原任务步长变成显式参数。
    外层 24 组常驻核编号不能直接替代原任务分片编号或原 block_num。
    此函数不提交新任务，也不拥有跨任务通知/等待。
    """
    t_dim = pl.tensor.dim(freqs_sin, 0)
    rope_sign_blocks = (t_dim + ROPE_ROWS - 1) // ROPE_ROWS
    for rope_rb in pl.range(logical_worker, rope_sign_blocks, logical_workers):
        rope_t0 = rope_rb * ROPE_ROWS
        rope_rows = pl.min(ROPE_ROWS, t_dim - rope_t0)
        il_ones = pl.tile.full([ROPE_ROWS, ROPE_DIM], dtype=pl.FP32, value=1.0)
        il_lane_ids = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        il_col = pl.col_expand_mul(il_ones, il_lane_ids)
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        rope_sin_rows = pl.load(freqs_sin, [rope_t0, 0], [ROPE_ROWS, ROPE_DIM],
                                valid_shape=[rope_rows, ROPE_DIM])
        rope_sign_rows = pl.set_validshape(il_sign, rope_rows, ROPE_DIM)
        pl.store(pl.mul(rope_sin_rows, rope_sign_rows), [rope_t0, 0], idx_sin_signed)
    return idx_sin_signed


@pl.jit.inline(auto_scope=False)
def cache_writeback_body(
    kv: pl.Tensor[[TOKENS, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[TOKENS, 2], pl.INT32],
    kv_cache_flat: pl.InOut[pl.Tensor[[CACHE_ROWS, HEAD_DIM], pl.BF16]],
    logical_worker: pl.Scalar[pl.INDEX],
    logical_workers: pl.Scalar[pl.INDEX],
):
    """来自冻结 baseline decode_csa.py:395；保留 device slot 和无效 slot 判断。

    页缓存的 GM reshape 与原代码一样在 InCore 外完成；这里接收二维视图。
    否则把四维动态 Tensor 的 reshape 放入核内会被解释为整张 tile 搬运。
    """
    t_dim = pl.tensor.dim(kv, 0)
    wb_blocks = (t_dim + WRITE_ROWS - 1) // WRITE_ROWS
    for wb_blk in pl.range(logical_worker, wb_blocks, logical_workers):
        wb_t0 = wb_blk * WRITE_ROWS
        for write_dt in pl.range(pl.min(WRITE_ROWS, t_dim - wb_t0)):
            write_t = wb_t0 + write_dt
            write_page = pl.read(ori_slot_mapping, [write_t, 0])
            write_offset = pl.read(ori_slot_mapping, [write_t, 1])
            if write_page >= 0 and write_offset >= 0:
                write_row = pl.cast(write_page, pl.INDEX) * BLOCK_SIZE + write_offset
                kv_cache_flat[write_row:write_row + 1, 0:HEAD_DIM] = kv[write_t:write_t + 1, 0:HEAD_DIM]
    return kv_cache_flat


@pl.jit
def vector_bodies_probe(
    freqs_sin: pl.Tensor[[TOKENS, ROPE_DIM], pl.FP32],
    idx_sin_signed: pl.Out[pl.Tensor[[TOKENS, ROPE_DIM], pl.FP32]],
    kv: pl.Tensor[[TOKENS, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[TOKENS, 2], pl.INT32],
    kv_cache_flat: pl.InOut[pl.Tensor[[CACHE_ROWS, HEAD_DIM], pl.BF16]],
):
    # 仅验证两个真实计算体能进入同一次 SPMD。两者的输入在探针入口已经就绪；
    # 此处不包含 CSA 的 QA/metadata 前置，也不代表最终静态排程或 NPU 正确性。
    # 编译探针直接接收原缓存的二维 GM 视图，避开 JIT 对临时 reshape 参数的
    # metadata 推断限制；它不创建第二份缓存，也未改变生产 CSA 的四维入口。
    t_dim = pl.tensor.dim(freqs_sin, 0)
    rope_workers = pl.min((t_dim + ROPE_ROWS - 1) // ROPE_ROWS, 16)
    with pl.spmd(24, name_hint="extracted_csa_vector_bodies"):
        worker = pl.tile.get_block_idx()
        if worker < rope_workers:
            idx_sin_signed = rope_sign_body(freqs_sin, idx_sin_signed, worker, rope_workers)
        if worker < 8:
            # 显式反向传入原分片号，确认 helper 不再暗中读取外层物理 worker。
            kv_cache_flat = cache_writeback_body(kv, ori_slot_mapping, kv_cache_flat, 7 - worker, 8)
    return idx_sin_signed, kv_cache_flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"device_execution": False, "complete_static_csa": False,
              "bodies": ["csa_rope_sign", "csa_cache_writeback"],
              "tokens_dynamic": True, "cache_rows_dynamic": True}
    try:
        compiled = vector_bodies_probe.compile(config=RunConfig(
            platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "kernel"),
            dump_passes=True))
        result.update(code_generation_completed=True, output_dir=str(compiled.output_dir))
    except Exception as error:
        result.update(code_generation_completed=False, error=str(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
