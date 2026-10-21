# SPDX-License-Identifier: Apache-2.0
"""TND模型图的捕获合同：重放前只比较Host描述符、策略和输入地址。

同址Q/L/positions内容可以变化，Tensor值不进入签名。请求槽容量、根输入
描述符、页存储地址或策略改变时，旧图不能被直接重放。捕获到Native/空路径的
图也不能假定后续仍走同一路径，统一回到正常Python派发重新判定。
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TNDGraphContract:
    runtime: Any
    token_capacity: int
    signature: tuple | None

    def allows_replay(self, context) -> bool:
        if self.signature is None:
            return False
        return self.runtime.graph_signature(context, self.token_capacity) == self.signature


def tensor_descriptor(value):
    if value is None:
        return None
    return (
        tuple(value.shape),
        value.stride(),
        value.dtype,
        value.device,
        value.data_ptr(),
    )
