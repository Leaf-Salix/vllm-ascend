# SPDX-License-Identifier: Apache-2.0
"""在 CSA 的精度版与性能版之间选择实现。

两套算子并存：性能版（`deepseek_v4_flash_csa`）改用 pypto-lib 上游的数值
写法以追平其性能，精度版（`deepseek_v4_flash_dspark`）以 Native 的规约、量化次序为准。
逐元素差异以验证记录为准，不承诺所有浮点状态逐 bit 一致。

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
_PERFORMANCE = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"
_ENV = "PTO_CSA_VARIANT"
_ALIASES = {"precision": "precision", "prec": "precision", "performance": "performance", "perf": "performance"}


_BISECT_PREFIX = "pkg:"


def csa_runtime() -> str:
    from vllm_ascend import envs

    runtime = envs.PTO_CSA_RUNTIME
    if runtime not in ("tensormap_and_ringbuffer", "host_build_graph"):
        raise ValueError(f"Unsupported PTO_CSA_RUNTIME={runtime!r}")
    return runtime


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
        name = value[len(_BISECT_PREFIX) :].strip()
        if not name.isidentifier():
            raise ValueError(f"{_ENV}={value!r} 的包名不合法")
        return f"vllm_ascend.ops.pypto.{name}"
    return _PRECISION if selected_variant() == "precision" else _PERFORMANCE


_RING_CONFIG_KEY = "pto_csa_ring_config"
_RING_HEAP_ENV = "PTO_CSA_RING_HEAP_MB"
_RING_TASK_WINDOW_ENV = "PTO_CSA_RING_TASK_WINDOW"


def _per_ring(value, source: str, scale: int = 1):
    """把标量或 4 元序列规范成 `pypto.torch.init` 接受的形状。

    字符串按逗号切分，便于同一份解析既吃 `additional_config` 的列表，也吃环境变量。
    """
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    if isinstance(value, (list, tuple)):
        parts = [int(v) * scale for v in value]
    else:
        parts = [int(value) * scale]
    if len(parts) not in (1, 4):
        raise ValueError(f"{source} 需要 1 个或 4 个值（a2a3 共 4 个 scope-depth ring），得到 {len(parts)} 个")
    if any(v <= 0 for v in parts):
        raise ValueError(f"{source} 的每一项都必须是正数，得到 {parts}")
    return parts[0] if len(parts) == 1 else tuple(parts)


def ring_sizing_kwargs(additional_config=None) -> dict:
    """解析 PyPTO 运行时 arena 的尺寸，返回 `pypto.torch.init` 的关键字参数。

    来源有两处，环境变量优先（它是取证用的临时覆盖，应当能压过常驻配置）：

        additional_config={"pto_csa_ring_config": {"heap_mb": [256, 128, 256, 32],
                                                   "task_window": 4096}}
        PTO_CSA_RING_HEAP_MB=256,128,256,32
        PTO_CSA_RING_TASK_WINDOW=4096

    `heap_mb` 是每个 scope-depth ring 的 heap（MiB），`task_window` 是每个 ring 的
    任务窗口深度。两者都可以给一个值（广播到全部 4 个 ring）或给 4 个值逐 ring 指定。
    都不给时返回空 dict，`pypto.torch.init` 就落到 Simpler 的编译期默认，
    行为与引入本开关之前完全一致。

    ## 为什么只能在 init 时给

    kernel 模式下运行时 arena 由 `prepare_kernel_runtime_impl` 在 `pypto.torch.init`
    期间按 `CallConfig.runtime_env` 一次建好并冻结，之后的 `prepare_callable` /
    `launch` 都不再携带 CallConfig。a2a3 的编译期默认是每 ring 256 MiB heap、
    16384 深 task window、共 4 个 ring，`pypto.torch.init()` 因此固定占用
    1.343 GiB 设备显存，与 kernel、batch、层数都无关——这是 PTO 相对 Native
    多出的 non-torch 显存的全部来源。

    ## 为什么不透出 ring_dep_pool

    `pypto.torch.init` 还接受 `ring_dep_pool`，但 A3 实测它对显存**没有可测影响**
    （heap 固定时 0.7957 vs 0.7962 GiB，在噪声内），开一个不起作用的旋钮只会误导人。

    ## 代价

    heap 给小了不是静默劣化，而是 dispatch 期 `Task Allocator Deadlock` 整进程退出；
    task window 不足同样会触发这个检测器。而且下限跟形状走：同一套 CSA kernel，
    ring 1 在 batch4 / history8K 下只要 50 MiB，在 batch24 / history128K 下要
    74.7 MiB。所以**这里不设默认值**，档位必须按目标形状实测后显式给。

    已实测可用的一组（DSV4 Flash 128K / B24 / EP16，16 卡）：
    `heap_mb=[256, 128, 256, 32]` + `task_window=4096`，省 0.574 GiB，
    使可用 KV 达到 23.07 GiB、24 路并发零抢占，且 forward 时延无劣化。
    取证入口见 `<output>/ascend/debug/device-N/device-*.log`，heap 耗尽时会直接写明
    `Heap ring N: used=… / Requested: … bytes`。
    """
    config = (additional_config or {}).get(_RING_CONFIG_KEY) or {}
    if not isinstance(config, dict):
        raise ValueError(f"additional_config[{_RING_CONFIG_KEY!r}] 必须是 dict，得到 {type(config).__name__}")
    unknown = set(config) - {"heap_mb", "task_window"}
    if unknown:
        raise ValueError(f"additional_config[{_RING_CONFIG_KEY!r}] 不认识的键：{sorted(unknown)}")

    kwargs = {}
    raw = os.environ.get(_RING_HEAP_ENV, "").strip() or config.get("heap_mb")
    if raw is not None and raw != "":
        source = _RING_HEAP_ENV if os.environ.get(_RING_HEAP_ENV, "").strip() else f"{_RING_CONFIG_KEY}.heap_mb"
        kwargs["ring_heap"] = _per_ring(raw, source, 1024 * 1024)
    raw = os.environ.get(_RING_TASK_WINDOW_ENV, "").strip() or config.get("task_window")
    if raw is not None and raw != "":
        source = (
            _RING_TASK_WINDOW_ENV
            if os.environ.get(_RING_TASK_WINDOW_ENV, "").strip()
            else f"{_RING_CONFIG_KEY}.task_window"
        )
        kwargs["ring_task_window"] = _per_ring(raw, source)
    return kwargs
