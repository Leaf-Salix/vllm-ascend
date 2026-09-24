# SPDX-License-Identifier: Apache-2.0
"""从精度版重导出：本模块不含任何数值实现，两套必须完全一致。

缓存布局常量由 Native 的存储格式决定，与选哪套算子无关。


"""

from ..deepseek_v4_flash_dspark.layout import *  # noqa: F401,F403
from ..deepseek_v4_flash_dspark import layout as _source

__all__ = [name for name in dir(_source) if not name.startswith("_")]

# 本集成不使用 pl.set_cache_policy(..., BYPASS)。上游 56e879c 给每个 decode 权重矩阵
# 都加了 BYPASS，但那条提交的标题是 "stream every DeepSeek V4 weight **NZ-ordered**
# and uncached"——绕过缓存是和 NZ 分块布局一起引入的，而本项目 NZ 硬性关闭
# （weight_nz_mode=0 / VLLM_ASCEND_ENABLE_NZ=0），权重是 ND 且来自 vLLM 的分配器。
# 2026-09-24 实测：加上 BYPASS 后设备侧必崩，且有两种形态——
#   decode_indexer.py 的两处（wq_b / weights_proj）直接触发 aicore 异常 507015，
#     故障日志写明 "The DDR address of the MTE instruction is out of range"；
#   其余几处不报本地故障，但 16 个 rank 同时 507057（远端错误），是集合通信不一致。
# 全部撤掉后（bis_bisG）exit=0、泳道正常采到。要重新启用必须先开 NZ 并单独验证。
