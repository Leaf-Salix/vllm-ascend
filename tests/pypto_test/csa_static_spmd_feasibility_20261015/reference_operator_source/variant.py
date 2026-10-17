# SPDX-License-Identifier: Apache-2.0
"""CSA默认且唯一维护性能实现；精度版已于2026-10-01封存。

保留performance/perf兼容值，以及排队实验所需的性能版私有包选择。
精度版历史源码和复现说明见tests/pypto_test/archive/csa_precision_20261001。
该维护决策不改变已有Native/PTO精度未对齐的结论。
"""

import os

_CSA_PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"
_ENV = "PTO_CSA_VARIANT"


_BISECT_PREFIX = "pkg:"


def csa_runtime() -> str:
    from vllm_ascend import envs

    runtime = envs.PTO_CSA_RUNTIME
    if runtime not in ("tensormap_and_ringbuffer", "host_build_graph"):
        raise ValueError(f"Unsupported PTO_CSA_RUNTIME={runtime!r}")
    return runtime


def selected_variant() -> str:
    """验证选择后返回performance；不将旧precision配置静默改名。"""
    variant_package()
    return "performance"


def variant_package() -> str:
    """定位实现包。

    `PTO_CSA_VARIANT=pkg:<name>` 直接指定 `vllm_ascend.ops.pypto.<name>`，用于把同一
    份性能版拆成几个只差一处改动的副本、同时排进队列做定位。副本必须与性能版同构
    （连同layout / native_storage / service_config整包复制），且只在定位期间
    存在，定位完就删，不作为长期形态。
    """
    from vllm_ascend import envs

    value = envs.PTO_CSA_VARIANT.strip()
    if value.lower() in ("precision", "prec", "pkg:deepseek_v4_flash_dspark"):
        raise ValueError(
            f"{_ENV}={value!r}: CSA精度版已封存；请取消该配置或使用performance。"
            "历史复现见tests/pypto_test/archive/csa_precision_20261001/README.md"
        )
    if value.lower().startswith(_BISECT_PREFIX):
        name = value[len(_BISECT_PREFIX):].strip()
        if not name.isidentifier() or name == "deepseek_v4_flash_dspark":
            raise ValueError(f"{_ENV}={value!r} 的包名不合法")
        return f"vllm_ascend.ops.pypto.{name}"
    if value.lower() not in ("performance", "perf"):
        raise ValueError(f"{_ENV}只接受performance/perf或性能版实验pkg:<name>，得到{value!r}")
    return _CSA_PACKAGE


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
        source = (_RING_TASK_WINDOW_ENV if os.environ.get(_RING_TASK_WINDOW_ENV, "").strip()
                  else f"{_RING_CONFIG_KEY}.task_window")
        kwargs["ring_task_window"] = _per_ring(raw, source)
    return kwargs
