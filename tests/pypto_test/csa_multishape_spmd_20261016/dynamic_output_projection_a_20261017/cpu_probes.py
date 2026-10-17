# SPDX-License-Identifier: Apache-2.0
"""三个独立动态编译壳：ND/NZ O_A与分组量化，不执行设备或模拟完整CSA。

token_axis只提供真实T的动态描述符；实际融合从已有业务Tensor取同样的dim。
packed、FP32/INT8与scale容量分别独立。ND用N128、NZ用N256覆盖两种列策略；
quant选RB96以保留一般尾零循环。每壳24/48worker仅检查源级动态分工和后端，
没有外层生产者，所以不宣称完成跨阶段同步或完整CSA。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.output_projection_a_stages import (
    MM_ROWS_DYN,
    PACK_TOTAL_ROWS_DYN,
    SCALE_ROWS_DYN,
    o_projection_a_stage_nd,
    o_projection_a_stage_nz,
    o_projection_group_quant_stage,
)

T_DYN = pl.dynamic("CSA_OA_PROBE_T")


@pl.jit(auto_scope=False)
def nd_probe(
    packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16],
    token_axis: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.InOut[pl.Tensor[[MM_ROWS_DYN, 8192], pl.FP32]],
):
    for worker in pl.spmd(24):
        output = o_projection_a_stage_nd(packed, weight, output, pl.tensor.dim(token_axis, 0), worker, 24, 128)
    return output


@pl.jit(auto_scope=False)
def nz_probe(
    packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ],
    token_axis: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.InOut[pl.Tensor[[MM_ROWS_DYN, 8192], pl.FP32]],
):
    for worker in pl.spmd(24):
        output = o_projection_a_stage_nz(packed, weight, output, pl.tensor.dim(token_axis, 0), worker, 24, 256)
    return output


@pl.jit(auto_scope=False)
def quant_probe(
    raw: pl.Tensor[[MM_ROWS_DYN, 8192], pl.FP32],
    token_axis: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.InOut[pl.Tensor[[MM_ROWS_DYN, 8192], pl.INT8]],
    scale: pl.InOut[pl.Tensor[[8, SCALE_ROWS_DYN], pl.FP32]],
):
    for worker in pl.spmd(48):
        output, scale = o_projection_group_quant_stage(raw, output, scale, pl.tensor.dim(token_axis, 0), worker, 48, 96)
    return output, scale
