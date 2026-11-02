# SPDX-License-Identifier: Apache-2.0
"""T192 WO-A：保留M128/N256父任务归属，隔离N子块、K分块及原生核归属的起点。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.output_projection_a_stages import o_projection_a_stage_nz


@pl.jit.inline(auto_scope=False)
def k80_arm(
    packed: pl.Tensor[[2048, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ],
    projected: pl.InOut[pl.Tensor[[256, 8192], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    REORDER: pl.constexpr,
):
    # Native最终baseK=80、stepK=4：13个L1窗口，最后一个仅256列。
    for logical in pl.range(worker, 64, 24):
        group = logical // 8
        row = logical // 4 % 2 * 128
        valid = pl.min(128, 192 - row)
        for sub in pl.unroll(2):
            column = (logical % 4 * 2 + sub) * 128
            first_panel = pl.cast(0, pl.INDEX)
            if REORDER:
                native_logical = group * 8 + column // 128
                native_owner = native_logical // 3
                if native_logical >= 48:
                    native_owner = 16 + (native_logical - 48) // 2
                first_panel = native_owner % 13 * 4
            first_k = pl.max(first_panel, 0) * 80
            a = pl.slice(packed, [128, 80], [group * 256 + row, first_k], valid_shape=[valid, 80])
            b = pl.slice(weight, [1, 80, 128], [group, first_k, column], valid_shape=[1, 80, 128])
            acc = pl.matmul(a, b, out_dtype=pl.FP32)
            for step in pl.pipeline(1, 52, stage=2):
                k = (pl.max(first_panel, 0) + step) % 52 * 80
                valid_k = pl.min(80, 4096 - k)
                a = pl.slice(packed, [128, 80], [group * 256 + row, k], valid_shape=[valid, valid_k])
                b = pl.slice(weight, [1, 80, 128], [group, k, column], valid_shape=[1, valid_k, 128])
                acc = pl.matmul_acc(acc, a, b)
            projected = pl.assemble(projected, acc, [row, group * 1024 + column])
    return projected


@pl.jit
def woa192_arm(
    packed: pl.Tensor[[2048, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ],
    projected: pl.InOut[pl.Tensor[[256, 8192], pl.FP32]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(24, name_hint="woa192_same_owner", allow_early_resolve=True):
        if MODE == 0:
            projected = o_projection_a_stage_nz(packed, weight, projected, 192, worker, 24, 256, 128)
        elif MODE == 3 or MODE == 4:
            projected = k80_arm(packed, weight, projected, worker, MODE == 4)
        else:
            for logical in pl.range(worker, 8 * 2 * 4, 24):
                group = logical // 8
                row = logical // 4 % 2 * 128
                valid = pl.min(128, 192 - row)
                for sub in pl.unroll(2):
                    column = logical % 4 * 2 + sub
                    native_logical = group * 8 + column
                    native_owner = native_logical // 3
                    if native_logical >= 48:
                        native_owner = 16 + (native_logical - 48) // 2
                    first_panel = pl.cast(0, pl.INDEX)
                    if MODE == 2:
                        # 借用已测T240的24核分工映射；T192仍须独立验收。
                        first_panel = native_owner % 8 * 2
                    first_k = pl.max(first_panel, 0) * 256
                    a = pl.slice(packed, [128, 256], [group * 256 + row, first_k], valid_shape=[valid, 256])
                    b = weight[group : group + 1, first_k : first_k + 256, column * 128 : column * 128 + 128]
                    acc = pl.matmul(a, b, out_dtype=pl.FP32)
                    for panel in pl.range(pl.max(first_panel, 0) + 1, 16):
                        k = panel * 256
                        chunk_a = pl.slice(packed, [128, 256], [group * 256 + row, k], valid_shape=[valid, 256])
                        chunk_b = weight[group : group + 1, k : k + 256, column * 128 : column * 128 + 128]
                        acc = pl.matmul_acc(acc, chunk_a, chunk_b)
                    for wrapped in pl.range(pl.max(first_panel, 0)):
                        k_wrap = wrapped * 256
                        wrap_a = pl.slice(packed, [128, 256], [group * 256 + row, k_wrap], valid_shape=[valid, 256])
                        wrap_b = weight[group : group + 1, k_wrap : k_wrap + 256, column * 128 : column * 128 + 128]
                        acc = pl.matmul_acc(acc, wrap_a, wrap_b)
                    projected = pl.assemble(projected, acc, [row, group * 1024 + column * 128])
    return projected
