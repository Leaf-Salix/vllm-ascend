# SPDX-License-Identifier: Apache-2.0
"""小M WKV按Native累加分档；保留原N128 owner和所有外层同步。

当前小M候选在M16边界用自然顺序，其他M<64用paired顺序。
T240独立采用已验收的按列K256次序；其余大M和独立joint-BF16路径沿用已有实现。
"""

import pypto.language as pl

from .kv_native_stages import kv_order_stage, kv_t240_stage
from .kv_stages import BF16_WEIGHT_LAYOUT, PAD_ROWS_DYN, T_DYN, kv_project_stage


@pl.jit.inline(auto_scope=False)
def kv_precision_project_stage(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    if tokens == 240:
        projected = kv_t240_stage(x, weight, projected, worker, WORKERS)
    elif tokens < 64:
        projected = kv_order_stage(x, weight, projected, worker, WORKERS, tokens % 16 != 0)
    else:
        projected = kv_project_stage(x, weight, projected, worker, WORKERS)
    return projected
