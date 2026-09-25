# SPDX-License-Identifier: Apache-2.0
"""权重 NZ 布局的开关，取自 vllm-ascend 统一的那一套。

`weight_nz_mode` 的语义与 `vllm_ascend/ascend_config.py` 一致：
0 = 关闭，1 = 只有量化（INT8）权重走 NZ（默认），2 = BF16/FP16 权重也走 NZ。

这里读的是环境变量 `VLLM_ASCEND_ENABLE_NZ`，而不是 `AscendConfig.weight_nz_mode`，
原因是 kernel 的参数布局是**模块加载时**由类型注解定下来的，而 AscendConfig 要等
vllm 初始化完才拿得到；环境变量在进程启动时就固定了。两者的默认值与取值语义相同，
并且 `native_adapter.prepare_weights` 会拿 AscendConfig 的值和这里做一致性校验——
布局标注与主机侧打包必须由同一个开关驱动，否则 kernel 读到的字节次序就是错的。

注意 NZ 分支对 kernel 写法有额外要求：切片偏移必须能被证明非负、且行偏移是 16 的
倍数、列偏移是一条 C0 线的倍数。所以并非所有权重都能简单地打开——目前只有
`wo_a`（proj_a_mm）改好了，其余仍走 ND，与本开关无关。
"""

import os

import pypto.language as pl

WEIGHT_NZ_MODE = int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1"))

# BF16/FP16 权重是否按 NZ 分形序存放
BF16_WEIGHT_NZ = WEIGHT_NZ_MODE >= 2
# INT8 量化权重是否按 NZ 分形序存放。注意这一档在 vllm-ascend 的默认值（1）下就是
# 开的——也就是说 wq_b / wo_b 走 NZ 不需要把 mode 调到 2，与 BF16 权重不同。
QUANT_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1

# 直接放进 pl.Tensor 的第三个槽：None 等价于不声明 layout（即 ND）。
# PyPTO 支持把 layout 放在闭包变量里，见 pypto/python/pypto/jit/cache.py 的说明。
BF16_WEIGHT_LAYOUT = pl.NZ if BF16_WEIGHT_NZ else None
QUANT_WEIGHT_LAYOUT = pl.NZ if QUANT_WEIGHT_NZ else None
