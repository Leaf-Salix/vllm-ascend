# SPDX-License-Identifier: Apache-2.0
"""仅T24：先复现原WO-A，再同M16/N128比较自然序与历史半程旋转。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.output_projection_a_stages import o_projection_a_stage_nz


@pl.jit
def woa_arm(
    packed: pl.Tensor[[1024, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ],
    projected: pl.InOut[pl.Tensor[[128, 8192], pl.FP32]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(24, name_hint="woa_same_input", allow_early_resolve=True):
        if MODE == 0:
            projected = o_projection_a_stage_nz(packed, weight, projected, 24, worker, 24, 128, 128)
        else:
            for logical in pl.range(worker, 8 * 2 * 8, 24):
                group = pl.max(0, logical // 16)
                row_block = logical // 8 % 2
                column = pl.max(0, logical % 8)
                row = row_block * 16
                valid = pl.min(16, 24 - row)
                first_panel = 0
                if MODE == 2:
                    # 旧精度T24表的假设，不宣称是本工具链的Native契约。
                    band = 0
                    if column >= 3:
                        if column < 6:
                            band = 1
                    first_panel = (group % 2 + band) % 2 * 8
                first_k = pl.max(first_panel, 0) * 256
                first_x = pl.slice(packed, [16, 256], [group * 128 + row, first_k], valid_shape=[valid, 256])
                first_w = weight[group : group + 1, first_k : first_k + 256, column * 128 : column * 128 + 128]
                acc = pl.matmul(first_x, first_w, out_dtype=pl.FP32)
                for panel in pl.range(pl.max(first_panel, 0) + 1, 16):
                    k = panel * 256
                    chunk_x = pl.slice(packed, [16, 256], [group * 128 + row, k], valid_shape=[valid, 256])
                    chunk_w = weight[group : group + 1, k : k + 256, column * 128 : column * 128 + 128]
                    acc = pl.matmul_acc(acc, chunk_x, chunk_w)
                for wrapped_panel in pl.range(pl.max(first_panel, 0)):
                    wrapped_k = wrapped_panel * 256
                    wrapped_x = pl.slice(packed, [16, 256], [group * 128 + row, wrapped_k], valid_shape=[valid, 256])
                    wrapped_w = weight[
                        group : group + 1, wrapped_k : wrapped_k + 256, column * 128 : column * 128 + 128
                    ]
                    acc = pl.matmul_acc(acc, wrapped_x, wrapped_w)
                projected = pl.assemble(projected, acc, [row, group * 1024 + column * 128])
    return projected
