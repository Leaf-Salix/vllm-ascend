# SPDX-License-Identifier: Apache-2.0
"""权重 NZ 布局的开关——实现在精度版，这里只重导出。

与 `hc_pre` / `hc_post` / `rmsnorm` / `layout` / `native_storage` / `service_config`
同一种做法：精度版是「源」，性能版重导出，保证两版由**同一个**开关驱动。开关一旦
分叉，kernel 的 layout 标注与主机侧的字节次序就可能对不上——那种错不报错、只算错。
"""

from ..deepseek_v4_flash_dspark.nz_mode import *  # noqa: F401,F403
from ..deepseek_v4_flash_dspark.nz_mode import (  # noqa: F401
    BF16_WEIGHT_LAYOUT,
    WO_A_WEIGHT_LAYOUT,
    BF16_WEIGHT_NZ,
    QUANT_WEIGHT_LAYOUT,
    QUANT_WEIGHT_NZ,
    WEIGHT_NZ_MODE,
)
