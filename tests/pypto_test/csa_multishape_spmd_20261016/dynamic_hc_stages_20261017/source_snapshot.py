# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""完整动态 CSA 的 HC 核内阶段；保留原算术，不创建独立任务。

这里是可维护的高层 PyPTO 源码，不是固定 B16 的展开 IR。T 从输入张量的
动态轴读取；显式 group/vector worker 决定每核范围，所有波次在核内循环。
所有整块 GM reshape 均由外层提供二维描述符视图；本核仅 reshape 已加载的 Tile。
本文件没有 pl.spmd、submit、全核屏障或核间通知。完整 CSA 的调度层负责：
HC_INPUT → HC_PARTIAL → HC_RAW → {HC_GATES、HC_COMB}，HC_GATES → HC_NORM。
发布须在该核 payload 写回完成后，消费者须等待对应 producer 并维护数据缓存。

首版保留原六个数学阶段和 GM 中间量，供新的单 SPMD 控制层嵌入；没有宣称
已完成尾段融合或性能优化。FP32 投影输入、BF16 norm 权重与所有舍入点不变。
原数学取自相邻 deepseek_v4_flash_csa/{decode_csa,hc_pre,rmsnorm}.py。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.rmsnorm import rms_norm_apply, rms_norm_inverse

T_DYN = pl.dynamic("HC_STAGE_T")
PAD_ROWS_DYN = pl.dynamic("HC_STAGE_PAD16_ROWS")
PARTIAL_ROWS_DYN = pl.dynamic("HC_STAGE_PARTIAL_ROWS")
MIX_ROWS_DYN = pl.dynamic("HC_STAGE_PAD8_ROWS")

# DSV4 Flash 模型维度及已验证数学分块；这些不是 batch/sequence 白名单。
D = 4096
HC_MULT = 4
HC_DIM = HC_MULT * D
HC_DIM_INV = 1.0 / HC_DIM
MIX_HC = 24
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
COMB_T_TILE = T_TILE
LINEAR_T_TILE = 16
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_K_TILE = 256
RMS_K_TILE = 512
D_TILE = 256
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6


def workspace_contract(tokens):
    """Host 只按已有 shape 描述分配空间，不读取设备元素。

    tokens 是本次实际 T；若使用容量缓冲，应传下列实际形状的连续子视图。
    group_workers/vector_workers 必须为正，worker 为零起点且小于对应数量。
    所有核按相同分块公式分工；一个 token 行块只归属一个 worker。
    四份 tail scratch 各仅被全局唯一的最后短块写入，不能用于改变行分工后
    多个 worker 同时拥有短块的方案；并发执行整层必须拥有独立工作区。
    """
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("HC 工作区要求正整数实际 token 数")
    padded16 = (tokens + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    padded8 = (tokens + T_TILE - 1) // T_TILE * T_TILE
    return {
        "x_fp32": {"shape": [tokens, HC_DIM], "dtype": "FP32"},
        "inv_rms": {"shape": [padded16, 1], "dtype": "FP32"},
        "widen_tail": {"shape": [T_TILE, HC_DIM], "dtype": "FP32"},
        "mixes_partials": {"shape": [LINEAR_OK * padded16, MIX_PAD], "dtype": "FP32"},
        "mixes_raw": {"shape": [padded16, MIX_PAD], "dtype": "FP32"},
        "pre_val_store": {"shape": [padded16, HC_PAD], "dtype": "FP32"},
        "post_tail_store": {"shape": [T_TILE, HC_PAD], "dtype": "FP32"},
        "comb_tail_store": {"shape": [COMB_T_TILE, HC_PAD * HC_MULT], "dtype": "FP32"},
        "x_mixed": {"shape": [padded8, D], "dtype": "BF16"},
        "x_normed_tail_store": {"shape": [T_TILE, D], "dtype": "BF16"},
    }


@pl.jit.inline(auto_scope=False)
def hc_widen_rms_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    x_fp32: pl.Out[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    inv_rms: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[T_TILE, HC_DIM], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：BF16 无损扩展并按原 32 个 K512 顺序累加输入平方和。

    调用方在本核返回后发布 HC_INPUT；AIC 投影及后续门值消费者等待所需行。
    尾行显式恢复 valid_shape 并清零后归约，避免 cast 丢失有效行标记。"""
    t_dim = pl.tensor.dim(x, 0)
    x_hc_flat = x
    x_hc32_flat = x_fp32
    widen_rows = (t_dim + T_TILE - 1) // T_TILE
    for wave in pl.range((widen_rows + vector_workers - 1) // vector_workers):
        widen_blk = vector_worker + wave * vector_workers
        if widen_blk < widen_rows:
            w_t0 = widen_blk * T_TILE
            w_rows = pl.min(T_TILE, t_dim - w_t0)
            w_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for w_db in pl.pipeline(HC_MULT * D // RMS_K_TILE, stage=4):
                w_d0 = w_db * RMS_K_TILE
                w_src = pl.slice(x_hc_flat, [T_TILE, RMS_K_TILE], [w_t0, w_d0], valid_shape=[w_rows, RMS_K_TILE])
                w_val = pl.cast(w_src, pl.FP32)
                if w_rows == T_TILE:
                    x_hc32_flat[w_t0 : w_t0 + T_TILE, w_d0 : w_d0 + RMS_K_TILE] = w_val
                    w_sq = pl.mul(w_val, w_val)
                    w_sq_row = pl.reshape(pl.row_sum(w_sq), [1, T_TILE])
                    w_sq_sum = pl.add(w_sq_sum, w_sq_row)
                else:
                    w_valid = pl.set_validshape(w_val, w_rows, RMS_K_TILE)
                    w_clean = pl.fillpad(w_valid, pad_value=pl.PadValue.zero)
                    w_sq_tail = pl.mul(w_clean, w_clean)
                    w_sq_row_tail = pl.reshape(pl.row_sum(w_sq_tail), [1, T_TILE])
                    w_sq_sum = pl.add(w_sq_sum, w_sq_row_tail)
                    widen_tail[0:T_TILE, w_d0 : w_d0 + RMS_K_TILE] = w_val
                    w_out = pl.load(
                        widen_tail,
                        [0, w_d0],
                        [T_TILE, RMS_K_TILE],
                        valid_shape=[w_rows, RMS_K_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    pl.store(w_out, [w_t0, w_d0], x_hc32_flat)
            w_mean = pl.add(pl.mul(w_sq_sum, HC_DIM_INV), NORM_EPS)
            w_inv = pl.reshape(pl.rsqrt(w_mean, high_precision=True), [T_TILE, 1])
            inv_rms[w_t0 : w_t0 + T_TILE, 0:1] = w_inv
    return x_fp32, inv_rms


@pl.jit.inline(auto_scope=False)
def hc_project_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    mixes_partials: pl.Out[pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32]],
    group_worker: pl.Scalar[pl.INDEX],
    group_workers: pl.Scalar[pl.INDEX],
):
    """AIC：M16、四段 K4096，每段仍按原 16 次 K256 顺序累加。

    调用前等待 HC_INPUT，确保所有本核输入行的 FP32 写回完成；返回后发布
    HC_PARTIAL。逻辑 task=(行块×4+K段)，多波次属于同一组核，不另派任务。"""
    t_dim = pl.tensor.dim(x, 0)
    t_linear = (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    x_flat = x
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    for wave in pl.range((linear_units + group_workers - 1) // group_workers):
        task = group_worker + wave * group_workers
        if task < linear_units:
            t0 = task // LINEAR_OK * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_linear_chunk = pl.slice(
                    x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE]
                )
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=kb == 0)
            partial_row0 = linear_split * t_linear + t0
            mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc
    return mixes_partials


@pl.jit.inline(auto_scope=False)
def hc_reduce_range(
    mixes_partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    mixes_raw: pl.Out[pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：原四份投影按 ((p0+p1)+p2)+p3 顺序归约。

    调用前等待所需四段 HC_PARTIAL；返回后发布 HC_RAW。按完整 M16
    归约含 padding 的输出行，不把四份左折叠改成树形 col_sum。"""
    t_linear = pl.tensor.dim(mixes_raw, 0)
    linear_blocks = t_linear // LINEAR_T_TILE
    for wave in pl.range((linear_blocks + vector_workers - 1) // vector_workers):
        linear_block = vector_worker + wave * vector_workers
        if linear_block < linear_blocks:
            linear_t0 = linear_block * LINEAR_T_TILE
            mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            for linear_split in pl.range(1, LINEAR_OK):
                partial_t0 = linear_split * t_linear + linear_t0
                partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
                mixes_total = pl.add(mixes_total, partial_tile)
            mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total
    return mixes_raw


@pl.jit.inline(auto_scope=False)
def hc_gates_range(
    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Out[pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_tail_store: pl.InOut[pl.Tensor[[T_TILE, HC_PAD], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：保留原 pre/post sigmoid、scale 乘法和 eps 顺序。

    调用前等待 HC_INPUT 与 HC_RAW；返回后发布 HC_GATES，供 mix/norm
    和最终 HCpost 消费。pre 按八列物理布局保留，post 只有四列有效。"""
    t_dim = pl.tensor.dim(post, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
        ob = vector_worker + wave * vector_workers
        if ob < token_tiles:
            t0 = ob * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            inv_col = inv_rms[t0 : t0 + T_TILE, 0:1]
            pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
            pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
            pre_val = pl.add(pre_sig, HC_EPS)
            pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val
            post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
            post_scaled = pl.mul(
                pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv_col), scale1
            )
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == T_TILE:
                post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
                    post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT]
                )
            else:
                post_tail_store[0:T_TILE, 0:HC_PAD] = post_pad
                post_tile = pl.load(
                    post_tail_store,
                    [0, 0],
                    [T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(post_tile, [t0, 0], post)
    return pre_val_store, post


@pl.jit.inline(auto_scope=False)
def hc_comb_range(
    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base_2d: pl.Tensor[[1, MIX_HC], pl.FP32],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
    comb_tail_store: pl.InOut[pl.Tensor[[COMB_T_TILE, HC_PAD * HC_MULT], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：CSA 原除法路径的二十轮 Sinkhorn，保持行列归约树。

    调用前等待 HC_INPUT 与 HC_RAW；与 hc_gates_range/mix 独立，返回后
    发布 HC_COMB。不引入 HCA 的 recip 替代路径，不改变迭代次数。"""
    t_dim = pl.tensor.dim(comb, 0)
    token_tiles = (t_dim + COMB_T_TILE - 1) // COMB_T_TILE
    for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
        ob = vector_worker + wave * vector_workers
        if ob < token_tiles:
            scale2 = pl.read(hc_scale, [2])
            t0 = ob * COMB_T_TILE
            valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
            inv_col_t = pl.load(
                inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec
            )
            comb_off = HC_MULT * 2
            mix_g0 = pl.load(
                mixes_raw,
                [t0, comb_off + 0 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            mix_g1 = pl.load(
                mixes_raw,
                [t0, comb_off + 1 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            mix_g2 = pl.load(
                mixes_raw,
                [t0, comb_off + 2 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            mix_g3 = pl.load(
                mixes_raw,
                [t0, comb_off + 3 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb0 = pl.load(
                hc_base_2d,
                [0, comb_off + 0 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb1 = pl.load(
                hc_base_2d,
                [0, comb_off + 1 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb2 = pl.load(
                hc_base_2d,
                [0, comb_off + 2 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb3 = pl.load(
                hc_base_2d,
                [0, comb_off + 3 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
            row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
            row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
            row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
            row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
            row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
            row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
            row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)
            row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row0_max = pl.row_max(row0_p, row_max_tmp)
            row1_max = pl.row_max(row1_p, row_max_tmp)
            row2_max = pl.row_max(row2_p, row_max_tmp)
            row3_max = pl.row_max(row3_p, row_max_tmp)
            row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
            row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
            row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
            row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
            row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
            row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
            row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
            row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
            row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)
            row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
            row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
            row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
            row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
            row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
            row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
            row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
            row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)
            row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_eff, col_sum)
            row1_cur = pl.div(row1_eff, col_sum)
            row2_cur = pl.div(row2_eff, col_sum)
            row3_cur = pl.div(row3_eff, col_sum)
            for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
                row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
                row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
                row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
                row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
                row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
                col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                col_sum = pl.add(col_sum, HC_EPS)
                row0_cur = pl.div(row0_norm, col_sum)
                row1_cur = pl.div(row1_norm, col_sum)
                row2_cur = pl.div(row2_norm, col_sum)
                row3_cur = pl.div(row3_norm, col_sum)
            if valid_rows == COMB_T_TILE:
                row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
                row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
                row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
                row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
                pl.store(row0_out, [t0, 0 * HC_MULT], comb)
                pl.store(row1_out, [t0, 1 * HC_MULT], comb)
                pl.store(row2_out, [t0, 2 * HC_MULT], comb)
                pl.store(row3_out, [t0, 3 * HC_MULT], comb)
            else:
                pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
                pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
                pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
                pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
                row0_tail = pl.load(
                    comb_tail_store,
                    [0, 0 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                row1_tail = pl.load(
                    comb_tail_store,
                    [0, 1 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                row2_tail = pl.load(
                    comb_tail_store,
                    [0, 2 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                row3_tail = pl.load(
                    comb_tail_store,
                    [0, 3 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
                pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
                pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
                pl.store(row3_tail, [t0, 3 * HC_MULT], comb)
    return comb


@pl.jit.inline(auto_scope=False)
def hc_mix_norm_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    pre_val_store: pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, D], pl.BF16]],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_normed_tail_store: pl.InOut[pl.Tensor[[T_TILE, D], pl.BF16]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：四路混合、BF16 舍入和完整 D4096 RMSNorm，原序不变。

    调用前等待 HC_GATES 以及本核输入行；返回后发布 HC_NORM。每行始终
    由一个 worker 完成全部 D，保留16个K256平方和的顺序，不新增跨核RMS。
    norm_w 直接接 Native BF16，不在初始化时转换为 FP32。"""
    t_dim = pl.tensor.dim(x, 0)
    x_flat = x
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
        token_block = vector_worker + wave * vector_workers
        if token_block < token_tiles:
            t0 = token_block * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            pre_tile = pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD]
            pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
            pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
            pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
            pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
            pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for mix_db in pl.pipeline(D // D_TILE, stage=2):
                d0 = mix_db * D_TILE
                x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
                x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, D + d0], valid_shape=[valid_rows, D_TILE])
                x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
                x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
                y0 = pl.row_expand_mul(x0, pre0)
                y1 = pl.row_expand_mul(x1, pre1)
                y2 = pl.row_expand_mul(x2, pre2)
                y3 = pl.row_expand_mul(x3, pre3)
                y01 = pl.add(y0, y1)
                y23 = pl.add(y2, y3)
                y = pl.add(y01, y23)
                y_bf16 = pl.cast(y, pl.BF16, mode="rint")
                x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_bf16
                y_rounded = pl.cast(y_bf16, pl.FP32)
                y_sq = pl.mul(y_rounded, y_rounded)
                y_sq_sum = pl.row_sum(y_sq)
                y_sq_row = pl.reshape(y_sq_sum, [1, T_TILE])
                sq_sum = pl.add(sq_sum, y_sq_row)
            norm_sq_sum = pl.create_tensor([1, T_TILE], dtype=pl.FP32)
            norm_sq_sum[:, :] = sq_sum
            y_inv = pl.create_tensor([1, T_TILE], dtype=pl.FP32)
            inverse_result = rms_norm_inverse(norm_sq_sum)
            y_inv[:, :] = inverse_result
            for norm_db in pl.pipeline(D // D_TILE, stage=2):
                d0 = norm_db * D_TILE
                mixed_input = x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE]
                norm_w_input = norm_w[d0 : d0 + D_TILE]
                norm_w_row = pl.reshape(norm_w_input, [1, D_TILE])
                normed_bf16 = rms_norm_apply(mixed_input, norm_w_row, y_inv)
                if valid_rows == T_TILE:
                    x_normed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = normed_bf16
                else:
                    x_normed_tail_store[0:T_TILE, d0 : d0 + D_TILE] = normed_bf16
                    normed_valid = pl.load(
                        x_normed_tail_store,
                        [0, d0],
                        [T_TILE, D_TILE],
                        valid_shape=[valid_rows, D_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    pl.store(normed_valid, [t0, d0], x_normed)
    return x_mixed, x_normed
