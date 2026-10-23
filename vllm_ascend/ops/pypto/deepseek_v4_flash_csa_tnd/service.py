# SPDX-License-Identifier: Apache-2.0
"""原生 TND 多 task 服务入口，复用数值无关的 Native/图/缓存生命周期。

输入及回退合同与 TND SPMD 相同；权重按本入口签名准备，资源策略和实际
提交使用本包。真实 Q/L 仍由设备 task 消费，不新增 Host 逐请求计划。
"""

from ..deepseek_v4_flash_csa.native_adapter import prepare_weights
from ..deepseek_v4_flash_csa_spmd_tnd.service import CSAServiceRuntime as TNDServiceRuntime
from .binding import PreparedDynamicCSA
from .entry import decode_csa_tnd
from .policy import select_policy


class CSAServiceRuntime(TNDServiceRuntime):
    @staticmethod
    def _prepare_weights(attention, layer):
        return prepare_weights(attention, None, layer, root_function=decode_csa_tnd)

    @staticmethod
    def _select_policy(*, compressed_table_columns=None, **descriptors):
        # 复用服务已有页表形状描述符，不读取设备Q/L或页表内容。
        return select_policy(compressed_table_columns=compressed_table_columns, **descriptors)

    @staticmethod
    def _prepare_binding(native, **options):
        return PreparedDynamicCSA(native, **options)
