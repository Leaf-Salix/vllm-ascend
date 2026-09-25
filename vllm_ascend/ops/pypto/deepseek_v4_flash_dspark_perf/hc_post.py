# SPDX-License-Identifier: Apache-2.0
"""从精度版重导出：本模块不含任何数值实现，两套必须完全一致。


mHC 的 pre/post 与 RMSNorm 片段是纯数值实现，性能版与精度版共用一份，
避免两套在门控、sinkhorn 迭代或归一化路径上分叉。
"""

from ..deepseek_v4_flash_dspark.hc_post import *  # noqa: F401,F403
from ..deepseek_v4_flash_dspark import hc_post as _source

__all__ = [name for name in dir(_source) if not name.startswith("_")]
