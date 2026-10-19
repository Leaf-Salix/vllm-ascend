### 问题

在 A2/A3 上，两个不同物理 Mat Tile 在分支之后仍然各自存活时，把其中一个选择为共同消费者的输入，会被 PyPTO 转换为不支持的 Mat→Mat `pto.tmov`，导致 PTOAS 拒绝编译。

这与 #3016 的“两个分支本来就指向同一个窗口”不同：本例两个输入 **必须保留独立存储**。不能把修复简化为合并输入地址或删除必要拷贝。也不是 #2902 的已处于 Mat 的 matmul 输入被重复 staging。

### 复现与已验证环境

- 在独立、tracked diff 为空的上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 自行构建后，以下完整最小程序通过公开 `pypto.ir.compile.compile` 接口复现。
- 当前 CSA 使用的独立调试构建（以 `d01de8523` 为本地基点并包含已有局部修复）也复现。不是只根据旧日志推断。
- 后端 `a2a3`、`MemoryPlanner.PYPTO`；CPU 完整 lowering/codegen 即可，无需模型、权重或 NPU。
- 使用的 PTOAS 为本地 v3（基于 `b7b4a82c` 的隔离构建）。错误的 `Mat→Mat` 操作已存在于 PyPTO 输出的 PTO IR 中，PTOAS 的目标约束拒绝本身是正确的。
- 不声称已经用最新 main 的完整独立构建重复验证。

保存为 `mat_phi_public_repro.py`，在已构建的 PyPTO 环境执行：

```bash
python mat_phi_public_repro.py /tmp/mat_phi_repro_output
```

```python
# SPDX-License-Identifier: Apache-2.0
"""两个仍存活的 Mat 缓冲之间做条件选择；不依赖 gm_view、MGATHER 或 CSA。"""
import json
import sys
import traceback
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print
from pypto.runtime import CompileOptions


@pl.jit.incore
def mat_phi_core(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [0, 0], out)
    # a/b 在选择后仍分别使用，不能通过把它们强制合并成同一物理地址来修复。
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [16, 0], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [32, 0], out)
    return out


@pl.jit
def mat_phi(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = mat_phi_core(x, y, flag, out)
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(compiled=False, device_execution=False)
    try:
        (folder / "symbolic.py").write_text(python_print(mat_phi.specialize()))
        from pypto.ir.compile import compile as compile_program
        artifact = compile_program(mat_phi.specialize(), platform="a2a3", output_dir=str(folder / "kernel"),
                                   dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
        result.update(compiled=True, artifact=str(artifact.output_dir))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False))
    raise SystemExit(0 if result["compiled"] else 1)
```

### 实际结果

```
'pto.tmov' op expects a supported tmov address-space pair for this target
```

最小生成 IR 中 `a` 的 Mat 地址为 0，`b` 为 512，`rhs` 为 1024，`selected` 又指向 0；else 分支生成从 b 到地址 0 的 Mat→Mat TMOV。选择之后的 `keep_a`、`keep_b` 仍分别消费原 a、b。

因此除编译失败外，该地址选择还需要检查存活性：**即使未来某后端支持 Mat→Mat 拷贝，也不能覆盖后续仍需要的 a**。本次在 PTOAS 验证阶段即失败，未执行这种潜在错误路径。

### 预期及修改建议

1. 对仍存活的不同输入 Mat 缓冲，分支结果应表达所选存储的引用/地址，或者将其消费者下沉到各分支；不要默认用数值拷贝实现所有 Mat phi。
2. 若后端需要物化分支结果，MemoryReuse/YieldFixup 必须尊重输入在分支后仍存活的事实，且先判断目标架构是否支持该存储空间的复制。不支持时给出前端可理解的能力诊断，而不是产生已知非法 PTO IR。
3. 可先采用消费者下沉作为局部降低方案：

```python
if pl.read(flag, [0]) > 0:
    prod_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(prod_a, [0, 0], out)
else:
    prod_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(prod_b, [0, 0], out)
# a、b仍保持原存储，可继续独立消费。
```

CSA 的三槽 GM→L1 探针用上述写法已通过完整 CPU 编译；没有修改 PyPTO 组件源码或提交修改 PR。该绕行不等于通用编译器修复，也不构成完整 CSA 性能证明。

建议回归同时验证两个 flag 分支，输出应依次为 selected@rhs、a@rhs、b@rhs；另外覆盖两个输入都需要跨循环携带的情况。
