# SPDX-License-Identifier: Apache-2.0
"""从精度版重导出：本模块不含任何数值实现，两套必须完全一致。

缓存布局常量由 Native 的存储格式决定，与选哪套算子无关。


"""

from ..deepseek_v4_flash_dspark import layout as _source
from ..deepseek_v4_flash_dspark.layout import *  # noqa: F401,F403

__all__ = [name for name in dir(_source) if not name.startswith("_")]

# Cache policy is specified at the corresponding kernel load sites. ND/NZ
# variants now use BYPASS selectively; the old global "NZ disabled/no BYPASS"
# description no longer describes the retained implementation.
