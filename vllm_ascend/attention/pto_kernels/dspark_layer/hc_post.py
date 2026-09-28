# SPDX-License-Identifier: Apache-2.0
"""从上游 pypto-lib `models/deepseek_v4_flash_dspark/hc_post.py` 移植。

mHC 的收尾：y[t, out_h] = post[t, out_h] * x[t] + sum_in_h comb[t, in_h, out_h] * residual[t, in_h]。
其中只有 `post * x` 这一项依赖 attention 的输出，comb 与 residual 的组合
完全可以和 attention 并行——这正是把它搬进 PTO kernel 的理由。

与上游一样按 dtype 说明：上游 hc 残差流是 FP32 端到端，这里改成与
vllm-ascend Native 一致的 BF16 收发，在行一级 cast 成 FP32 参与计算。
prefill 入口与 standalone 用例不搬，本包只做 decode。
"""

import pypto.language as pl

from .config import DECODE_BATCH, DECODE_SEQ, TP
from .config import FLASH as M

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim

# tiling
T_TILE = 4
INACTIVE_FILL_T_TILE = 16
INACTIVE_FILL_D_TILE = 256
assert (DECODE_BATCH // TP * DECODE_SEQ) % T_TILE == 0


def _hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for token_block in pl.spmd(token_tiles, name_hint="hc_post"):
        t0 = token_block * T_TILE
        for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
            if t < t_dim:
                # One cast per token: all HC_MULT outputs share the x row.
                x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
                for out_h in pl.unroll(HC_MULT):
                    post_w = pl.read(post, [t, out_h])
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.pipeline(HC_MULT, stage=4):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_d = in_h * D
                        # 本包的 hc 残差流是 BF16（与 Native 的 npu_hc_post 一致），读出后抬成 FP32 再累加。
                        res_row = pl.cast(residual_flat[t : t + 1, res_d : res_d + D], pl.FP32)
                        weighted = pl.mul(res_row, comb_w)
                        y_row = pl.add(y_row, weighted)
                    # 写回下一层的 hc 残差流：与 Native 一致按 BF16 落盘。
                    y_flat[t : t + 1, out_h * D : out_h * D + D] = pl.cast(y_row, pl.BF16, mode="rint")
    return y


hc_post = pl.jit.inline(_hc_post)
hc_post_test = pl.jit(_hc_post)
