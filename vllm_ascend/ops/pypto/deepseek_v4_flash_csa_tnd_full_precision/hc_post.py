# SPDX-License-Identifier: Apache-2.0
"""A3 HC-post沿用已核对的PTO独立Mul/Add实现。

参考csrc/moe/hc_post/op_kernel/hc_post_d_split.h::DoCompute：
先FP32 post*x，再依次加residual[0..3]*comb[0..3,out]，最后BF16 RINT。
不采用950/310的MulAddDst路径；实际安装的Native算子仍须同输入逐bit验证。
"""

from ..deepseek_v4_flash_csa_tnd_precision.hc_post import hc_post_test  # noqa: F401
