# SPDX-License-Identifier: Apache-2.0
"""权重 NZ 布局的开关，取自 vllm-ascend 统一的那一套。

`weight_nz_mode` 的语义与 `vllm_ascend/ascend_config.py` 一致：
0 = 关闭，1 = 只有量化（INT8）权重走 NZ（默认），2 = BF16/FP16 权重也走 NZ。

这里读的是环境变量 `VLLM_ASCEND_ENABLE_NZ`，而不是 `AscendConfig.weight_nz_mode`，
原因是 kernel 的参数布局是**模块加载时**由类型注解定下来的，而 AscendConfig 要等
vllm 初始化完才拿得到。模型配置检查与算子注册前均校验环境、AscendConfig 和已导入的
布局模式一致；导入后修改环境也会被拒绝。
布局标注与 Native 存储绑定必须由同一个开关驱动，否则 kernel 读到的字节次序就是错的。

注意 NZ 分支对 kernel 写法有额外要求：切片偏移必须能被证明非负、且行偏移是 16 的
倍数、列偏移是一条 C0 线的倍数。两版四张目标权重均声明可选 NZ，
真实布局读取根签名；四张根矩阵方向与 Native 加载后相同，wo_b 保留二维 [G*K, D]。
"""

import inspect

import pypto.language as pl

from vllm_ascend import envs

WEIGHT_NZ_MODE = envs.VLLM_ASCEND_ENABLE_NZ
if WEIGHT_NZ_MODE not in (0, 1, 2):
    raise ValueError(f"VLLM_ASCEND_ENABLE_NZ must be 0, 1 or 2, got {WEIGHT_NZ_MODE}")

# BF16/FP16 权重是否按 NZ 分形序存放。
#
# 这里用 >=1 而不是 >=2，因为 `weight_nz_mode` 实际控制的是两件**独立**的事：
#   ① Native 侧张量的 npu format 转换——`ops/linear.py` 的 `_should_trans_nz` 对 BF16
#      权重要求 mode==2，对 INT8（`w8a8_dynamic.py`）mode>=1 就转；
#   ② PTO 侧 `prepare_weights` 里对**自己那份 transpose 副本**做的 `_pack_nz`。
# 两者机制不同：PTO 不消费 Native 转过的 FRACTAL_NZ（`weight()` 里还会
# `npu_format_cast` 回 ND），而是自己按 pto-isa 的分形序重排一份私有副本。所以在
# 同一个 mode 值下让 PTO 的 BF16 权重也走 NZ，不改变与 Native 的对比口径——两侧
# 拿到的是同一份原始权重、同一个配置值，只是 PTO 内部多摆了一次字节。
# 这一份副本本来就存在（`transpose().contiguous()`），NZ 化不额外占显存。
BF16_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1
# INT8 量化权重是否按 NZ 分形序存放。注意这一档在 vllm-ascend 的默认值（1）下就是
# 开的——也就是说 wq_b / wo_b 走 NZ 不需要把 mode 调到 2，与 BF16 权重不同。
QUANT_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1

# 直接放进 pl.Tensor 的第三个槽：None 等价于不声明 layout（即 ND）。
# PyPTO 支持把 layout 放在闭包变量里，见 pypto/python/pypto/jit/cache.py 的说明。
BF16_WEIGHT_LAYOUT = pl.NZ if BF16_WEIGHT_NZ else None
QUANT_WEIGHT_LAYOUT = pl.NZ if QUANT_WEIGHT_NZ else None


def validate_weight_nz_mode(effective_mode: int) -> None:
    """只在初始化阶段检查，禁止配置或环境与已经固定的根布局分叉。"""
    environment_mode = envs.VLLM_ASCEND_ENABLE_NZ
    if effective_mode not in (0, 1, 2) or not (
        effective_mode == environment_mode == WEIGHT_NZ_MODE
    ):
        raise ValueError(
            "PTO CSA weight NZ mode mismatch: "
            f"AscendConfig={effective_mode}, environment={environment_mode}, "
            f"imported_layout={WEIGHT_NZ_MODE}; set VLLM_ASCEND_ENABLE_NZ "
            "to the requested weight_nz_mode before importing PTO kernels"
        )


def root_weight_layouts(root_function) -> dict[str, str]:
    """读取本仓根函数的实际注解；日志与存储绑定共用，避免另维护一份 NZ 名单。"""
    params = inspect.signature(root_function).parameters
    result = {}
    for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
        layout = params[name].annotation.layout
        if layout is None:
            result[name] = "ND"
        elif layout == pl.NZ:
            result[name] = "NZ"
        else:
            raise ValueError(f"Unsupported PTO CSA weight layout: {name}={layout}")
    return result


def root_weight_shapes(root_function) -> dict[str, tuple[int, ...]]:
    """从根签名读取 Native 矩阵几何，回放按显式来源迁移旧转置权重。"""
    params = inspect.signature(root_function).parameters
    return {name: tuple(params[name].annotation.shape) for name in root_weight_layouts(root_function)}
