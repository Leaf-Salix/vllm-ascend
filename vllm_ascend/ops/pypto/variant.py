# SPDX-License-Identifier: Apache-2.0
"""在 CSA 的精度版与性能版之间选择实现。

两套算子并存：性能版（`deepseek_v4_flash_dspark_perf`）改用 pypto-lib 上游的数值
写法以追平其性能，精度版（`deepseek_v4_flash_dspark`）与 Native 逐 bit 一致。

选择走环境变量而不是 `additional_config`：切换的用途是对照实验，环境变量最不侵入
生产配置结构，测试驱动也便于按轮次切换。

**默认是精度版**。不设该变量时走的就是与 Native 对齐的那一套，行为与引入本模块
之前一致；性能对照必须显式 `PTO_CSA_VARIANT=performance`。

默认值一度改为性能版，很快又改回（用户 2026-09-24 两次裁定）。改回的直接教训是：
性能版仍在开发中，而 `@pl.jit` 会在编译时重读 kernel 源文件，于是任何不显式指定
版本的排队任务都会加载正在被编辑的那个包——B=40 的功能验证就因此报
`AssertionError: specialize: no def named 'sparse_attn_csa_tp1' in its own source`
而作废。默认指向已定型的一套更安全。开发期间提交的任务仍应显式写明版本。
"""

import os

_PRECISION = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark"
_PERFORMANCE = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf"
_ENV = "PTO_CSA_VARIANT"
_ALIASES = {"precision": "precision", "prec": "precision",
            "performance": "performance", "perf": "performance"}


_BISECT_PREFIX = "pkg:"


def selected_variant() -> str:
    """返回 "precision" 或 "performance"；无法识别的取值必须报错而不是静默回退。"""
    value = os.environ.get(_ENV, "precision").strip().lower()
    if value.startswith(_BISECT_PREFIX):
        return "performance"
    if value not in _ALIASES:
        raise ValueError(f"{_ENV} must be one of {sorted(_ALIASES)}, got {value!r}")
    return _ALIASES[value]


def variant_package() -> str:
    """定位实现包。

    `PTO_CSA_VARIANT=pkg:<name>` 直接指定 `vllm_ascend.ops.pypto.<name>`，用于把同一
    份性能版拆成几个只差一处改动的副本、同时排进队列做定位。副本必须与性能版同构
    （layout / native_storage / service_config 仍是对精度版的再导出），且只在定位期间
    存在，定位完就删，不作为长期形态。
    """
    value = os.environ.get(_ENV, "precision").strip()
    if value.lower().startswith(_BISECT_PREFIX):
        name = value[len(_BISECT_PREFIX):].strip()
        if not name.isidentifier():
            raise ValueError(f"{_ENV}={value!r} 的包名不合法")
        return f"vllm_ascend.ops.pypto.{name}"
    return _PRECISION if selected_variant() == "precision" else _PERFORMANCE
