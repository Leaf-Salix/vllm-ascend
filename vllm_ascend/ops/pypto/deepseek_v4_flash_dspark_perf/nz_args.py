# SPDX-License-Identifier: Apache-2.0
"""单卡回放用的根入参预处理：把本包声明为 pl.NZ 的权重改存成 NZ 分形序。

整模型路径在 native_adapter.prepare_weights 里做同一件事，两边由同一个开关
（nz_mode）驱动。开关关闭时这里什么都不做，根入参保持 ND。
"""

from .native_adapter import _pack_nz
from .nz_mode import BF16_WEIGHT_NZ

# 参数名 -> 该参数在 NZ 开启时需要重排（只列 kernel 签名里用了 NZ layout 的）
BF16_NZ_PARAMS = ("wo_a",)


def pack_args(tensors: dict) -> dict:
    if not BF16_WEIGHT_NZ:
        return tensors
    result = dict(tensors)
    for name in BF16_NZ_PARAMS:
        if name in result:
            result[name] = _pack_nz(result[name])
    return result
