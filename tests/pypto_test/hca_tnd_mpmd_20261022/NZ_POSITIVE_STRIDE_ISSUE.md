## 问题

HCA TND MPMD 的输出流水实验中，合法 NZ 列切片在 CPU lowering 被拒绝。
分档入口调用 `@pl.jit.inline(auto_scope=False)` helper，helper 在 `pl.spmd` 外定义
`workers = 24`，核内 `pl.range(worker, 32, workers)`；列地址为
`max(0, unit % 32 * 128)`，所有实际值均在 `[0,3968]`，且按 INT8 NZ 的 C0 对齐。

仅把 range 的步长改为直接常量 `24`，两档均完整后端编译通过。
没有设备执行，没有读取或修改共享安装。这个问题阻断编译，不是已观测到的数值错误。

## 环境与证据边界

- A3/a2a3；PYPTO 内存规划器。
- PyPTO 本地 kernel 调试分支 `26e94e05627eb47710dcc214bae1774b8829df8a`，父提交
  `e891351e305e1b8ac70ffb9ffe60a243fc0420dc`；最新增量为官方编排逻辑表达式修复。
  该环境含此前 CSA 修复，尚未在未打补丁的最新 main 上单独复测，不声称已经验证所有分支。
- PTOAS 源码 `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`。失败发生在 PyPTO 的 NZ lowering，未到 PTOAS。
- 同脚本结果：`captured_step.compiled=false`，`literal_step.compiled=true`。
- 单层、无 inline 分档的更小表达两版都通过；需要保留下面的分档 helper 结构复现。

## 可直接运行的最小复现

保存为 `repro_nz_positive_stride.py`，在已安装的 PyPTO 编译环境中运行：

```bash
python repro_nz_positive_stride.py --output /tmp/nz_stride_new_output
```

输出目录必须不存在。仅编译，不需要 NPU 或模型权重；错误和每个 pass 的 IR 保存在输出目录。

```python
# SPDX-License-Identifier: Apache-2.0
"""纯CPU复现：SPMD outlining丢失编排侧正步长，合法NZ列地址被拒绝。

两版只有range的步长写法不同：编排局部变量与直接编译常量。
没有设备执行、权重或外部文件依赖；输出IR用于核查Simplify和outlining的顺序。
"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


TOKENS = pl.dynamic("NZ_STRIDE_T")


@pl.jit.inline(auto_scope=False)
def captured_step(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
    ROWS: pl.constexpr,
):
    workers = 24
    with pl.spmd(workers):
        worker = pl.tile.get_block_idx()
        for unit in pl.range(worker, 32, workers):
            col = pl.max(0, unit % 32 * 128)
            lhs = pl.load(a, [0, 0], [ROWS, 256], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(w, [0, col], [256, 128], target_memory=pl.MemorySpace.Mat)
            value = pl.matmul(lhs, rhs, out_dtype=pl.INT32)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                shard = pl.aiv_shard(value)
                pl.store(shard, [unit * ROWS, lane * 64], out)
    return out


@pl.jit.inline(auto_scope=False)
def literal_step(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
    ROWS: pl.constexpr,
):
    with pl.spmd(24):
        worker = pl.tile.get_block_idx()
        for unit in pl.range(worker, 32, 24):
            col = pl.max(0, unit % 32 * 128)
            lhs = pl.load(a, [0, 0], [ROWS, 256], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(w, [0, col], [256, 128], target_memory=pl.MemorySpace.Mat)
            value = pl.matmul(lhs, rhs, out_dtype=pl.INT32)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                shard = pl.aiv_shard(value)
                pl.store(shard, [unit * ROWS, lane * 64], out)
    return out



@pl.jit(auto_scope=False)
def captured_entry(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
):
    tokens = pl.tensor.dim(a, 0)
    if tokens <= 16:
        out = captured_step(a, w, out, 16)
    else:
        out = captured_step(a, w, out, 32)
    return out


@pl.jit(auto_scope=False)
def literal_entry(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
):
    tokens = pl.tensor.dim(a, 0)
    if tokens <= 16:
        out = literal_step(a, w, out, 16)
    else:
        out = literal_step(a, w, out, 32)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, entry in (("captured_step", captured_entry), ("literal_step", literal_entry)):
        try:
            entry.compile(config=CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                dump_passes=True, output_dir=str(args.output / name),
            ))
            result[name] = {"compiled": True}
        except Exception as exc:
            result[name] = {"compiled": False, "error": str(exc)}
    result["device_execution"] = False
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

## 实际错误

```text
NZ layout requires the slice offset on shape[-1] to be non-negative,
and this one cannot be proven to be.
Check failed: IsProvableNonNegative(offset, facts, &sign_budget)
include/pypto/ir/transforms/utils/tensor_view_semantics.h:594
```

## Pass 级观察和原因判断

在完整 HCA 失败包的 `15_after_FlattenTileNdTo2D.py` 中，生成的 InCore 函数仍收到
`workers: Scalar[INDEX]` 参数；循环为 `range(get_block_idx(), units, workers)`，
列偏移已经简化为 `(unit % 32) * 128`。源码中 `max(0, ...)` 在 outlining 之前已被消掉。

因此前面的简化使用了调用方能证明的正步长信息，但后面的 NZ 检查只看提取后的函数，
看不到这个事实；单独修改 max 的书写顺序无效。这个结论来自中间 IR，尚未修改或验证上游修复。

## 建议修改方向

1. 优先在 outlining/调用特化时保留标量常量，避免把已知 `24` 变成没有范围信息的普通参数。
2. 对确实动态的范围，跨函数保留可证明的非负/正步长约束，或在调用点做合法的常量传播。
3. 若前两项暂不支持，应避免提前删除下一阶段需要的边界保护；不要关闭 NZ 负偏移检查，
   也不要把真实 NZ Tensor 标成 ND 来绕过。
4. 回归覆盖：单函数与 inline helper、两档 constexpr 特化、局部正步长/直接常量、
   `max(0,offset)`；同时保留真正负偏移和未知符号步长的拒绝测试。

本地暂时直接在核内 range 使用编译常量24，保持原调度与数值。只提 issue 与建议，不提修复 PR。
