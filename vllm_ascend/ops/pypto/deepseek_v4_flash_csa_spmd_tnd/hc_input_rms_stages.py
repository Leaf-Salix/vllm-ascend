# SPDX-License-Identifier: Apache-2.0
"""HC原始16384维输入的Native-order RMS；不处理mixed后的4096维Norm。

平方partial沿K768分为22段（末段256），按Cube归约后取对角线；Vector按
段号左折叠，再执行sqrt和1/div。算法参考本项目已验证的precision_spmd
hc_prefix_stages；是否匹配本机Native必须由同输入设备对拍确定。
生产者发布和消费者等待由调用层负责；这里不创建任务或额外同步。
"""

import pypto.language as pl

from .hc_stages import HC_DIM, HC_DIM_INV, NORM_EPS, PAD_ROWS_DYN, T_DYN, T_TILE
from .runtime_abi import GROUPS, VECTORS

RMS_PARTS = 22
SQUARE_ROWS = pl.dynamic("HC_INPUT_SQUARE_ROWS")


@pl.jit.inline(auto_scope=False)
def hc_input_square_range(
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    sq_partials: pl.InOut[pl.Tensor[[SQUARE_ROWS, 16], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(sq_partials, 0) // RMS_PARTS
    sq_group = 1
    if t_dim >= 64:
        sq_group = 4
    sq_groups = (RMS_PARTS + sq_group - 1) // sq_group
    sq_units = t_linear // 16 * sq_groups
    for sq_unit in pl.range(worker, sq_units, GROUPS):
        sq_t0 = sq_unit // sq_groups * 16
        sq_part_base = sq_unit % sq_groups * sq_group
        sq_rows = pl.min(16, t_dim - sq_t0)
        for sq_group_item in pl.range(pl.min(sq_group, RMS_PARTS - sq_part_base)):
            sq_part = sq_part_base + sq_group_item
            sq_count = pl.min(768, HC_DIM - sq_part * 768) // 32
            native_sq_acc = pl.create_tensor([16, 16], dtype=pl.FP32)
            for sq_kb in pl.pipeline(0, sq_count, stage=2):
                sq_k = sq_part * 768 + sq_kb * 32
                sq_lhs = pl.slice(x_flat, [16, 32], [sq_t0, sq_k], valid_shape=[sq_rows, 32])
                native_sq_acc = pl.matmul_acc(native_sq_acc, sq_lhs, sq_lhs, b_trans=True, init_cond=sq_kb == 0)
            sq_out_row = sq_part * t_linear + sq_t0
            sq_partials[sq_out_row : sq_out_row + 16, 0:16] = native_sq_acc
    return sq_partials


@pl.jit.inline(auto_scope=False)
def hc_input_rms_range(
    sq_partials: pl.Tensor[[SQUARE_ROWS, 16], pl.FP32],
    inv_rms: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(inv_rms, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for rms_block in pl.range(worker, token_tiles, VECTORS):
        rms_t0 = rms_block * T_TILE
        rms_lanes = pl.arange(0, [1, T_TILE], dtype=pl.INT32)
        rms_rows = pl.add(rms_lanes, pl.cast(rms_t0, pl.INT32))
        rms_local = pl.add(rms_lanes, pl.cast(rms_t0 % 16, pl.INT32))
        rms_index = pl.add(pl.mul(rms_rows, 16), rms_local)
        rms_total = pl.gather(sq_partials, index=rms_index)
        for rms_part in pl.range(1, RMS_PARTS):
            rms_part_index = pl.add(rms_index, pl.cast(rms_part * t_linear * 16, pl.INT32))
            rms_value = pl.gather(sq_partials, index=rms_part_index)
            rms_total = pl.add(rms_total, rms_value)
        rms_variance = pl.add(pl.mul(rms_total, HC_DIM_INV), NORM_EPS)
        rms_root = pl.sqrt(rms_variance)
        native_inverse = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=1.0), rms_root)
        inv_rms[rms_t0 : rms_t0 + T_TILE, 0:1] = pl.reshape(native_inverse, [T_TILE, 1])
    return inv_rms
