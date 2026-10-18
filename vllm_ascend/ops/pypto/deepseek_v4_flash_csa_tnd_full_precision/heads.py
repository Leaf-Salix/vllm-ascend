# SPDX-License-Identifier: Apache-2.0
"""同一BF16数据的按group整行复制；不转换dtype，不做浮点运算。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.decode_csa import HEAD_DIM, O_GROUP_IN, O_GROUPS, T_DYN, T_PAD, H


@pl.jit.inline(auto_scope=False)
def publish_native_heads(
    source: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    target: pl.Out[pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16]],
    tokens: pl.Scalar[pl.INDEX],
    ready: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(48, name_hint="native_heads", deps=[ready]) as _export_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, tokens, 48):
            for group in pl.range(O_GROUPS):
                value = pl.load(source, [group * T_PAD + row, 0], [1, O_GROUP_IN])
                pl.store(value, [row, group * O_GROUP_IN], target)
    return target
