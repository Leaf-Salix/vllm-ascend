# 热eager调用重复进行JIT特化分析，HCA的Host提交约7ms

## 现象与范围

PyPTO `26e94e05627eb47710dcc214bae1774b8829df8a`，Python3.10、torch/torch_npu2.10、
A3、TMR kernel模式。相关Python JIT/torch源码在此本地提交未修改；不涉及ACLGraph replay。
同一个已经编译、预热和注册的算子，在每次普通torch.ops调用中仍重做较多JIT分析。
本问题按用户要求只记录和提出改进建议，当前不修改SDK、不提PR。

这不是#2881讨论的进程首次toolchain/磁盘cache查找，也不是#2629的DP8 host_orch TaskArgs构造。
这里是**同一进程热调用**，编译产物及Worker registration已存在。

## 实测

同进程无PyTorch profiler，7次计时中位数，单位μs：

| HCA 128K/B16/S6、36个Tensor参数 | Host提交 | 提交至同步完成 |
|---|---:|---:|
| Native | 2031.19 | 2094.53 |
| 正常已预绑定PTO torch.ops | 7076.17 | 7746.50 |
| PTO完整服务入口 | 7636.17 | 8266.01 |
| 复用已预热SDK registration，仅诊断 | 1043.04 | 1714.15 |
| 完整服务入口＋相同registration，仅诊断 | 1572.14 | 2220.07 |

对照沿用`pypto.torch.launch.enqueue(registration, args)`，保留CallSignature参数检查、
Tensor shape/dtype/布局及别名校验、当前设备/流、Worker/PID/generation存活检查和异步Tensor所有权。
它仅省掉重复JIT特化查找，未改变设备二进制、计算、调度或Tensor内容。HCA输出和全部cache/state
与普通入口逐bit一致，保护区通过；缺失参数仍被拒绝。该内部接口仅用于实验，未进入正式适配代码。

下面无模型权重的最小复现只执行一个加法kernel，根调用有32层inline转发链，2个Tensor入参。
同进程15次中位数：普通注册调用提交4263.746μs，prepared对照178.572μs；
含设备同步分别4476.959和381.574μs。新Tensor地址及动态T=16/32也通过，缺失参数被拒绝。

## cProfile定位

以下为HCA各3次cProfile调用累计除3的cumulative时间；**cProfile会增加开销，不能与上述无插桩时间相加或直接代换**：

| 函数 | μs/调用 |
|---|---:|
| JITFunction._resolve_compiled | 约11289 |
| _resolve_specialization / _bind_args | 约3102 / 3091 |
| _resolve_constexpr_bindings / _expand_constexpr_variants | 约2756 / 2749 |
| _get_source_hash | 约2168 |
| _compute_per_func_dyndim_maps | 约2025 |
| describe_eager_call | 约1764 |

位置：`python/pypto/jit/decorator.py`的`__call__`、`_resolve_compiled`、
`_resolve_specialization`、`_resolve_constexpr_bindings`以及`python/pypto/torch/launch.py`。
函数间存在包含关系，不能求和。预热registration命中位于上述特化/缓存键推导之后，
因此已注册不等于已经跳过重复Python图分析。

## 修改建议（不是已实现的API）

建议提供显式prepare/freeze后的kernel callable，例如下列**伪接口**：

```python
prepared = pypto.torch.prepare(kernel, example_args, config=compile_options)
# 后续仍传本轮Tensor和动态shape，仍走当前流与完整runtime所有权检查。
prepared(*current_args)
```

需要明确契约：

1. prepare固化源码、constexpr、静态维度、layout及编译配置；代码/常量热更新需要重新prepare，
   或通过有版本的guard失效，不能静默使用陈旧二进制。普通JIT的现有动态检查语义保持。
2. 每次调用仍验证当前Tensor的动态shape、dtype、设备、布局、storage bounds及alias；
   不缓存原始指针假装对象仍存活，不删除Worker/PID/generation及异步所有权检查。
3. 固定ABI的ParamInfo、CallSignature、dtype标签等可一次预解析，进一步减少Python对象重建；
   Tensor本轮元数据和当前stream仍逐调用获取，不能把上次CallFrame直接复用。
4. 同一已准备模板内允许动态T、新allocation和更新后的设备metadata；不要求每个Tensor对象固定。
5. 如继续优化普通JIT路径，可缓存不可变的维度传播/constexpr计划，并用明确依赖版本失效；
   不建议简单永久memoize所有全局常量或关闭正确性检查。

## 最小复现（独立文件，无模型权重）

运行需要已经安装的PyPTO/Simpler/PTOAS和torch_npu环境，设备通过本机正常调度分配：
`python repro.py --output /absolute/new_directory --device 0`。
第一次预热编译不纳入计时，脚本不修改SDK，不生成或重放ACLGraph。

```python
# SPDX-License-Identifier: Apache-2.0
"""无需模型权重的热JIT查找复现：一条32层inline链，最终只有一个加法kernel。

    python repro_eager_host_lookup.py --output /absolute/new_dir --device 0

普通注册调用与已预热registration使用相同二进制、相同输入和SDK参数校验。
后一条只作诊断，不作为生产公共API；所有源码、constexpr和编译配置保持不变。
"""

import argparse
import cProfile
import importlib.util
import json
import os
import pstats
import statistics
import sys
import time
from pathlib import Path


def write_graph(path, depth):
    source = '''import pypto.language as pl
T = pl.dynamic("T")

@pl.jit.inline
def leaf(x: pl.Tensor[[T, 32], pl.FP32], y: pl.Tensor[[T, 32], pl.FP32], bias: pl.constexpr):
    with pl.spmd(1):
        row0 = pl.tile.get_block_idx() * 32
        rows = pl.tensor.dim(x, 0)
        tile = pl.load(x, [row0, 0], [32, 32], valid_shape=[rows - row0, 32])
        pl.store(pl.add(tile, bias), [row0, 0], y)
    return y
'''
    previous = "leaf"
    for index in range(depth):
        name = f"stage_{index}"
        source += f'''
@pl.jit.inline
def {name}(x: pl.Tensor[[T, 32], pl.FP32], y: pl.Tensor[[T, 32], pl.FP32], bias: pl.constexpr):
    y = {previous}(x, y, bias + 1.0)
    return y
'''
        previous = name
    source += f'''
@pl.jit
def entry(x: pl.Tensor[[T, 32], pl.FP32], y: pl.Out[pl.Tensor[[T, 32], pl.FP32]]):
    y = {previous}(x, y, 0.0)
    return y
'''
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("eager_lookup_graph", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "0").split(",")[0]))
    parser.add_argument("--depth", type=int, default=32)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(root / "build")
    import torch
    import torch_npu
    import pypto.torch
    from pypto.runtime.kernel.context import get_process_kernel_state
    from pypto.torch.launch import enqueue

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    entry = write_graph(root / "graph.py", args.depth)
    registered = pypto.torch.register(entry, "jit_host_lookup_repro::add")
    x = torch.ones((32, 32), dtype=torch.float32, device=f"npu:{args.device}")
    y = torch.empty_like(x)
    for _ in range(3):
        registered(x, y)
        torch.npu.synchronize()
    state = get_process_kernel_state()
    bound = state.require_config()
    registrations = [state.find_specialization(k, bound) for k in state._specializations if k[0] is entry]
    assert len(registrations) == 1
    registration = registrations[0]

    def ordinary():
        registered(x, y)

    def prepared():
        enqueue(registration, (x, y))

    result = {"depth": args.depth, "warmup_calls": 3, "aclgraph_replay": False,
              "sdk_modified": False, "timing": {}}
    for name, run in (("registered", ordinary), ("prepared_diagnostic", prepared)):
        samples = []
        for _ in range(15):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            run()
            end_submit = time.perf_counter_ns()
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            samples.append({"submit_us": (end_submit - start) / 1000, "completion_us": (end - start) / 1000})
        assert torch.equal(y.cpu(), torch.full((32, 32), 1.0 + args.depth))
        profile = cProfile.Profile()
        for _ in range(3):
            profile.runcall(run)
            torch.npu.synchronize()
        stats = [{"file": file, "line": line, "function": fn, "calls": nc,
                  "own_us": tt * 1e6, "cumulative_us": ct * 1e6}
                 for (file, line, fn), (cc, nc, tt, ct, callers) in pstats.Stats(profile).stats.items()]
        result["timing"][name] = {
            "samples": samples, "submit_median_us": statistics.median(v["submit_us"] for v in samples),
            "completion_median_us": statistics.median(v["completion_us"] for v in samples),
            "cprofile_calls": 3, "cprofile": sorted(stats, key=lambda v: v["cumulative_us"], reverse=True),
        }
    # 相同registration仍然接受动态T变化和新Tensor；不依赖旧地址/旧Tensor对象缓存。
    for rows in [16, 32]:
        fresh = torch.full((rows, 32), 2.0, dtype=torch.float32, device=x.device)
        target = torch.empty_like(fresh)
        enqueue(registration, (fresh, target))
        torch.npu.synchronize()
        assert torch.equal(target.cpu(), torch.full((rows, 32), 2.0 + args.depth))
    try:
        enqueue(registration, (x,))
    except (ValueError, TypeError) as exc:
        result["missing_argument_rejected"] = str(exc)
    else:
        raise AssertionError("参数校验被绕过")
    result["passed"] = True
    result["dynamic_T_new_buffers_pass"] = True
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {a: v[a] for a in ("submit_median_us", "completion_median_us")}
                      for k, v in result["timing"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

```
