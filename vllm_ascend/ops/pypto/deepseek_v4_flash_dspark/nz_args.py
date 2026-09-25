# SPDX-License-Identifier: Apache-2.0
"""单卡回放用的根入参预处理：把本包声明为 pl.NZ 的权重改存成 NZ 分形序。

整模型路径在 native_adapter.prepare_weights 里做同一件事，两边由同一个开关
（nz_mode）驱动。开关关闭时这里什么都不做，根入参保持 ND。

两张表分开是因为两档开关不同：BF16 权重要 weight_nz_mode>=2，INT8 量化权重
mode>=1 就开（也就是 vllm-ascend 的默认档）。
"""

from .native_adapter import _pack_nz
from .nz_mode import BF16_WEIGHT_NZ, QUANT_WEIGHT_NZ

BF16_NZ_PARAMS = ("wo_a",)
QUANT_NZ_PARAMS = ()


def pack_args(tensors: dict) -> dict:
    result = dict(tensors)
    for names, enabled in ((BF16_NZ_PARAMS, BF16_WEIGHT_NZ), (QUANT_NZ_PARAMS, QUANT_WEIGHT_NZ)):
        if not enabled:
            continue
        for name in names:
            if name in result:
                result[name] = _pack_nz(result[name])
    return result
