# SPDX-License-Identifier: Apache-2.0
"""单卡回放用的根入参预处理：把本包声明为 pl.NZ 的权重改存成 NZ 分形序。

整模型路径在 native_adapter.prepare_weights 里做同一件事。两边必须保持一致，
否则单卡回放测的布局和整模型实际跑的不是一回事。
"""

from .native_adapter import _pack_nz

# 参数名 -> 需要按 NZ 重排（只列本包签名里标了 pl.NZ 的）
NZ_PARAMS = ("wo_a",)


def pack_args(tensors: dict) -> dict:
    result = dict(tensors)
    for name in NZ_PARAMS:
        if name in result:
            result[name] = _pack_nz(result[name])
    return result
