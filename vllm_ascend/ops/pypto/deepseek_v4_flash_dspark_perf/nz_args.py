# SPDX-License-Identifier: Apache-2.0
"""单卡回放用的根入参预处理：把本包声明为 pl.NZ 的权重改存成 NZ 分形序。

整模型路径在 native_adapter.prepare_weights 里做同一件事，两边由同一个开关
（nz_mode）驱动。开关关闭时这里什么都不做，根入参保持 ND。
"""

from .native_adapter import _pack_nz
from .nz_mode import BF16_WEIGHT_NZ, QUANT_WEIGHT_NZ

# 只列 kernel 签名里真的用了 NZ layout 的权重。两张表分开是因为两档开关不同：
# BF16 权重要 weight_nz_mode>=2，INT8 量化权重 mode>=1 就开（也就是默认档）。
# wq_a 暂不列入：它的 NZ kernel 版在当前 PyPTO 上编不过，见 qkv_proj_rope.q_proj_qa
BF16_NZ_PARAMS = ("wo_a",)
QUANT_NZ_PARAMS = ("wq_b",)


def pack_args(tensors: dict) -> dict:
    result = dict(tensors)
    groups = ((BF16_NZ_PARAMS, BF16_WEIGHT_NZ), (QUANT_NZ_PARAMS, QUANT_WEIGHT_NZ))
    for names, enabled in groups:
        if not enabled:
            continue
        for name in names:
            if name in result:
                result[name] = _pack_nz(result[name])
    return result
