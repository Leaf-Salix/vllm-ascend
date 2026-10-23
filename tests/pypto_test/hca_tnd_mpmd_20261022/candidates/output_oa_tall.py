# SPDX-License-Identifier: Apache-2.0
"""HCA O_A大T候选：M192/N128减少129～192行时的权重重复读取。

只改变O_A供数/任务划分，整token量化、O_B及HC_post完全沿用正式Native数学。
本诊断入口未加入最终策略选择，不进入生产；先比较完整层及逐bit精度。
T仍动态，沿有效行裁剪；数学tile不等于固定batch或固定query长度。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.decode_o_proj import (
    D, O_GROUPS, O_GROUP_IN, O_LORA, T_PAD, PROJ_A_ROW_TILE,
    PROJ_A_MM_N_TILE, PROJ_A_LARGE_N_TILE,
    PROJ_B_SMALL_T_TILE, PROJ_B_MEDIUM_T_TILE, PROJ_B_MM_T_TILE,
    INT8_AMAX_EPS, INT8_SCALE_MAX, proj_a_mm,
)
from ..deepseek_v4_flash_csa.hc_post import hc_post_block
from ..deepseek_v4_flash_csa.nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

T_DYN = pl.dynamic("HCA_O_POST_TOKENS")
HC_MULT = 4
HC_DIM = HC_MULT * D
O_WIDTH = O_GROUPS * O_LORA
OB_L1_K = 512
VECTORS = 48
SCALE_ROW = 16

@pl.jit.inline(auto_scope=False)
def _native_oa_all(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    weight: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    projected: pl.Tensor[[T_PAD, O_WIDTH], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    # 大T诊断：M192覆盖129～192有效行，N128使L0C保持96KiB。
    # 原T>96路径为M128/N256；T>128时重复读取每份权重两次。
    # 本候选将行块扩为192，让该区间权重只遍历一次，仍有64个逻辑Cube任务。
    row_blocks = (tokens + 191) // 192
    with pl.spmd(O_GROUPS * row_blocks * 8, name_hint="hca_oa_tall",
                 deps=[heads_dep], allow_early_resolve=True) as ready:
        unit = pl.tile.get_block_idx()
        group = pl.max(0, unit // (row_blocks * 8))
        row0 = (unit // 8 % row_blocks) * 192
        col0 = pl.max(0, unit % 8) * 128
        active = pl.min(192, tokens - row0)
        source_row = group * T_PAD + row0
        pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
        seed_storage = pl.create_tile([192, 128], dtype=pl.FP32,
                                      target_memory=pl.MemorySpace.Acc, compact=True)
        seed = pl.tile.set_validshape(seed_storage, active, 128)
        # L1双缓冲320KiB，L0A两槽48KiB、L0B两槽32KiB；完整K保留同一Acc。
        for outer, (acc_outer,) in pl.pipeline(O_GROUP_IN // 256, stage=2, init_values=(seed,)):
            k0 = outer * 256
            lhs = pl.load(packed, [source_row, k0], [192, 256],
                          valid_shape=[active, 256], target_memory=pl.MemorySpace.Mat)
            rhs_group = pl.load(weight, [group, k0, col0], [1, 256, 128],
                                target_memory=pl.MemorySpace.Mat)
            rhs = pl.reshape(rhs_group, [256, 128])
            for inner, (acc_inner,) in pl.pipeline(4, stage=2, init_values=(acc_outer,)):
                left = pl.tile.extract(lhs, 0, inner * 64, [192, 64], target_memory=pl.MemorySpace.Left)
                right = pl.tile.extract(rhs, inner * 64, 0, [64, 128], target_memory=pl.MemorySpace.Right)
                update = pl.tile.matmul_acc(acc_inner, left, right, init_cond=(outer == 0 and inner == 0))
                inner_done = pl.yield_(update)
            outer_done = pl.yield_(inner_done)
        pl.store(outer_done, [row0, group * O_LORA + col0], projected)
    return projected, ready


@pl.jit.inline(auto_scope=False)
def _native_output(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_WIDTH, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
    MM_ROWS: pl.constexpr,
    OA_COLUMNS: pl.constexpr,
    OB_COLUMNS: pl.constexpr,
    OB_SPLITS: pl.constexpr,
    OB_L0_K: pl.constexpr,
):
    tokens = pl.tensor.dim(residual, 0)
    projected = pl.create_tensor([T_PAD, O_WIDTH], dtype=pl.FP32)
    quantized = pl.create_tensor([T_PAD, O_WIDTH], dtype=pl.INT8)
    # 每个token独占64B，避免不同AIV短行scale写共享存储块；仅首元素用于消费。
    token_scales = pl.create_tensor([T_PAD, SCALE_ROW], dtype=pl.FP32)
    integer_output = pl.create_tensor([OB_SPLITS * T_PAD, D], dtype=pl.INT32)
    projected, oa_ready = _native_oa_all(packed, wo_a, projected, tokens, heads_dep)

    with pl.spmd(pl.min(tokens, VECTORS), name_hint="hca_native_token_quant",
                 deps=[oa_ready], allow_early_resolve=True) as quant_ready:
        for token in pl.range(pl.tile.get_block_idx(), tokens, VECTORS):
            raw = pl.load(projected, [token, 0], [1, O_WIDTH])
            native_bf16 = pl.cast(raw, pl.BF16, mode="rint")
            rounded = pl.reshape(pl.cast(native_bf16, pl.FP32), [O_GROUPS, O_LORA])
            # A3列向量需要32B对齐：先算8段的max，8x1恰为32B。再将每段
            # 复制到8列并沿行归约，得到8份相同的全行amax；全过程留在Vector。
            reduce_tmp = pl.create_tile([O_GROUPS, O_LORA], dtype=pl.FP32)
            segment_max = pl.row_max(pl.abs(rounded), reduce_tmp)
            replicated = pl.row_expand_mul(pl.tile.full([O_GROUPS, 8], dtype=pl.FP32, value=1.0), segment_max)
            all_max = pl.col_max(replicated)
            amax = pl.maximum(all_max, pl.tile.full([1, 8], dtype=pl.FP32, value=INT8_AMAX_EPS))
            numerator = pl.tile.full([1, 8], dtype=pl.FP32, value=INT8_SCALE_MAX)
            # 与Native一致分别计算量化乘数和反量化scale，不倒数回推另一个。
            multiplier = pl.div(numerator, amax)
            dequant_scale = pl.mul(amax, 1.0 / INT8_SCALE_MAX)
            scaled = pl.row_expand_mul(rounded, pl.reshape(multiplier, [O_GROUPS, 1]))
            integer = pl.cast(scaled, pl.INT32, mode="rint")
            half = pl.cast(integer, pl.FP16, mode="round")
            byte = pl.cast(half, pl.INT8, mode="trunc")
            pl.store(pl.reshape(byte, [1, O_WIDTH]), [token, 0], quantized)
            scale_row = pl.concat(dequant_scale, dequant_scale)
            pl.store(scale_row, [token, 0], token_scales)

    rows = (tokens + MM_ROWS - 1) // MM_ROWS
    with pl.spmd(rows * (D // OB_COLUMNS) * OB_SPLITS, name_hint="hca_native_ob_split",
                 deps=[quant_ready], allow_early_resolve=True) as ob_ready:
        unit = pl.tile.get_block_idx()
        split = unit % OB_SPLITS
        column_unit = unit // OB_SPLITS
        row0 = column_unit // (D // OB_COLUMNS) * MM_ROWS
        col0 = pl.max(0, column_unit % (D // OB_COLUMNS)) * OB_COLUMNS
        active = pl.min(MM_ROWS, tokens - row0)
        seed_storage = pl.create_tile([MM_ROWS, OB_COLUMNS], dtype=pl.INT32,
                                      target_memory=pl.MemorySpace.Acc, compact=True)
        seed = pl.tile.set_validshape(seed_storage, active, OB_COLUMNS)
        for outer, (outer_acc,) in pl.pipeline((O_WIDTH // OB_SPLITS) // OB_L1_K, stage=2, init_values=(seed,)):
            k0 = split * (O_WIDTH // OB_SPLITS) + outer * OB_L1_K
            lhs = pl.load(quantized, [row0, k0], [MM_ROWS, OB_L1_K],
                          valid_shape=[active, OB_L1_K], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(wo_b, [k0, col0], [OB_L1_K, OB_COLUMNS], target_memory=pl.MemorySpace.Mat)
            for inner, (inner_acc,) in pl.pipeline(OB_L1_K // OB_L0_K, stage=2, init_values=(outer_acc,)):
                offset = inner * OB_L0_K
                left = pl.tile.extract(lhs, 0, offset, [MM_ROWS, OB_L0_K],
                                       target_memory=pl.MemorySpace.Left)
                right = pl.tile.extract(rhs, offset, 0, [OB_L0_K, OB_COLUMNS],
                                        target_memory=pl.MemorySpace.Right)
                updated = pl.tile.matmul_acc(inner_acc, left, right, init_cond=(outer == 0 and inner == 0))
                done_inner = pl.yield_(updated)
            done_outer = pl.yield_(done_inner)
        pl.store(done_outer, [split * T_PAD + row0, col0], integer_output)

    residual_flat = pl.reshape(residual, [tokens, HC_DIM])
    output_flat = pl.reshape(output, [tokens, HC_DIM])
    with pl.spmd(pl.min(tokens, VECTORS), name_hint="hca_native_output_post",
                 deps=[ob_ready], allow_early_resolve=True):
        channels = pl.reshape(pl.load(wo_scale, [0], [D]), [1, D])
        for token in pl.range(pl.tile.get_block_idx(), tokens, VECTORS):
            # 两段在INT32域相加后才转FP32；与完整K8192累加严格等价。
            # 若先各自转FP32，再相加，会额外引入舍入边界，不能这样改。
            first = pl.load(integer_output, [token, 0], [1, D])
            if OB_SPLITS == 1:
                acc = pl.cast(first, pl.FP32)
            else:
                second = pl.load(integer_output, [T_PAD + token, 0], [1, D])
                acc = pl.cast(pl.add(first, second), pl.FP32)
            by_channel = pl.col_expand_mul(acc, channels)
            token_scale = pl.read(token_scales, [token, 0])
            value = pl.mul(by_channel, token_scale)
            # 保留Native O_B输出的BF16边界；HC_post仍按post*x及0→3残差左折叠。
            bf16 = pl.create_tensor([1, D], dtype=pl.BF16)
            bf16 = pl.cast(value, pl.BF16, mode="rint")
            output_flat = hc_post_block(bf16, residual_flat, post, comb, output_flat, token, 0, 1, 1, D)
    return output


@pl.jit.inline
def o_proj_hc_post(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_WIDTH, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(residual, 0)
    if tokens <= PROJ_B_SMALL_T_TILE:
        output = _native_output(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
                                PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, 128, 1, 256)
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        output = _native_output(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
                                PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, 128, 1, 256)
    else:
        output = _native_output(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
                                PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE, 256, 2, 128)
    return output
