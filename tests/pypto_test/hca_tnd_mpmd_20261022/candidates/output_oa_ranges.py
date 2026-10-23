# SPDX-License-Identifier: Apache-2.0
"""HCA TND输出尾链：Native量化语义与按token区间选择的Cube分块。

整体思路
--------
1. 按T选择O_A：33<=T<=96合为一个网格，每组5份N192加1份N64，共48个任务；
   其余T沿用八组原流水。候选尚在验证，不能据B16单档结果推广。输出先落BF16再量化。
   Native将八组投影结果拼成一个8192列token后取amax，不能各组独立量化。
2. 量化任务等齐O_A生产者；最多48个Vector按token分工，一个token独占64B尺度行。
   8段行max→8×8列max全程留在Vector，避免Scalar FP32 max的PTOAS helper缺陷。
3. O_B按token有效行数和输出列分工。T<=32用M32，33<=T<=96用M96，均为
   N128、完整K8192；T>96用M128、N256、K分两段。末块valid_shape只计算有效行。
   分两段减少激活重复供数，INT32域相加后才转FP32，不改变整数点积结果。
4. Vector按channel尺度、token尺度反量化并落BF16，随后融合四路HC_post。
   各阶段由Tensor/TaskId依赖排序，没有新增GM轮询或主机读取。

最坏整数绝对和8192*128*128=134217728，低于INT32上限。这里恢复Native的
可见舍入/量化边界，不能与历史八组FP32反量化求和版本要求输出逐bit相同。
动态T及Native布局不变；外部B、query长度由上游TND几何处理，本段不假设T=B*S。
精度证据和区间选择的正负性能结果见ops/pypto/HCA_TND_MPMD_OPTIMIZATION_LOG.md。
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
def _oa_column_tile(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    weight: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    projected: pl.Tensor[[T_PAD, O_WIDTH], pl.FP32],
    group: pl.Scalar[pl.INDEX],
    row0: pl.Scalar[pl.INDEX],
    col0: pl.Scalar[pl.INDEX],
    active: pl.Scalar[pl.INDEX],
    source_row: pl.Scalar[pl.INDEX],
    COLUMNS: pl.constexpr,
):
    seed_storage = pl.create_tile([96, COLUMNS], dtype=pl.FP32,
                                  target_memory=pl.MemorySpace.Acc, compact=True)
    seed = pl.tile.set_validshape(seed_storage, active, COLUMNS)
    # A和B的L1窗口分开推进；A一份512列供两份256列B使用。
    for ka, (a_acc,) in pl.pipeline(O_GROUP_IN // 512, stage=2, init_values=(seed,)):
        a_k0 = ka * 512
        a_l1 = pl.load(packed, [source_row, a_k0], [96, 512],
                       valid_shape=[active, 512], target_memory=pl.MemorySpace.Mat)
        for kb, (b_acc,) in pl.pipeline(2, stage=2, init_values=(a_acc,)):
            b_k0 = a_k0 + kb * 256
            b_group = pl.load(weight, [group, b_k0, col0], [1, 256, COLUMNS], valid_shape=[1, 256, COLUMNS],
                              target_memory=pl.MemorySpace.Mat)
            b_l1 = pl.reshape(b_group, [256, COLUMNS])
            # L0B两份24KiB、L0A两份12KiB；72KiB Acc保持完整K累加链。
            # 末份只计算/写回64列，不能让物理COLUMNS列盒覆盖下一组。
            for ki, (c_acc,) in pl.pipeline(4, stage=2, init_values=(b_acc,)):
                a_l0 = pl.tile.extract(a_l1, 0, kb * 256 + ki * 64, [96, 64],
                                      target_memory=pl.MemorySpace.Left)
                b_l0 = pl.tile.extract(b_l1, ki * 64, 0, [64, COLUMNS],
                                      target_memory=pl.MemorySpace.Right)
                updated = pl.tile.matmul_acc(c_acc, a_l0, b_l0,
                                             init_cond=(ka == 0 and kb == 0 and ki == 0))
                c_done = pl.yield_(updated)
            b_done = pl.yield_(c_done)
        a_done = pl.yield_(b_done)
    pl.store(a_done, [row0, group * O_LORA + col0], projected)
    return projected


@pl.jit.inline(auto_scope=False)
def _native_oa_all(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    weight: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    projected: pl.Tensor[[T_PAD, O_WIDTH], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    OA_COLUMNS: pl.constexpr,
):
    # 沿用Native MDL的两级供数，N改成192，每组6份（末份64列）。
    # T<=96时8组共48个任务，正好两轮24 Cube，避免N256的32任务形成8核尾波。
    # 现在下游采用整token尺度，必须等齐所有组；因此可合并原八个O_A发布任务。
    # 这与旧的分组量化/O_B提前开工链不同，不把旧版单点试验直接当作本版收益。
    row_blocks = (tokens + 95) // 96
    with pl.spmd(O_GROUPS * row_blocks * 6, name_hint="hca_oa_balanced_columns",
                 deps=[heads_dep], allow_early_resolve=True) as ready:
        pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
        unit = pl.tile.get_block_idx()
        group = pl.max(0, unit // (row_blocks * 6))
        row0 = (unit // 6 % row_blocks) * 96
        col0 = pl.max(0, unit % 6) * 192
        active = pl.min(96, tokens - row0)
        source_row = group * T_PAD + row0
        # NZ张量的物理N维保持静态。前五份192列，末份64列；均严格在原权重内。
        if unit % 6 == 5:
            projected = _oa_column_tile(packed, weight, projected, group, row0, col0, active, source_row, 64)
        else:
            projected = _oa_column_tile(packed, weight, projected, group, row0, col0, active, source_row, 192)
    return projected, ready


@pl.jit.inline(auto_scope=False)
def _oa_original(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    weight: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    projected: pl.Tensor[[T_PAD, O_WIDTH], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    OA_COLUMNS: pl.constexpr,
):
    producers = pl.array.create(O_GROUPS, pl.TASK_ID)
    oa_rows = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    with pl.manual_scope():
        for group in pl.parallel(O_GROUPS):
            projected, tid = proj_a_mm(
                packed, weight, projected, group, group * T_PAD, group * O_LORA,
                tokens, oa_rows, heads_dep, OA_COLUMNS, PIPELINE_OA=True,
            )
            producers[group] = tid
    ready = pl.system.task_dummy(deps=[producers])
    return projected, ready


def _make_native_output(oa_producer):
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
        oa_rows = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
        projected = pl.create_tensor([T_PAD, O_WIDTH], dtype=pl.FP32)
        quantized = pl.create_tensor([T_PAD, O_WIDTH], dtype=pl.INT8)
        # 每个token独占64B，避免不同AIV短行scale写共享存储块；仅首元素用于消费。
        token_scales = pl.create_tensor([T_PAD, SCALE_ROW], dtype=pl.FP32)
        integer_output = pl.create_tensor([OB_SPLITS * T_PAD, D], dtype=pl.INT32)
        projected, oa_ready = oa_producer(packed, wo_a, projected, tokens, heads_dep, OA_COLUMNS)

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


    return _native_output


_native_output_regular = _make_native_output(_oa_original)
_native_output_medium = _make_native_output(_native_oa_all)


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
        output = _native_output_regular(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
                                PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, 128, 1, 256)
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        output = _native_output_medium(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
                                PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, 128, 1, 256)
    else:
        output = _native_output_regular(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
                                PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE, 256, 2, 128)
    return output
