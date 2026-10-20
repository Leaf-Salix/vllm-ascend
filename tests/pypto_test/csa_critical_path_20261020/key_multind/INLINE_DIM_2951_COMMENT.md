补充一个相关的原接口复现变体，供确认修复覆盖范围：**两个维度均为动态、没有循环、只读 view 并写另一输出**，仍会在描述符 shape/stride 中留下空字段。

本地实际复现基线是 CSA 隔离树 `2dfa7f120ba5e577566046d920845aa241bfa7bb`（包含其它已记录的本地修复）；没有用本轮新增的显式 strides 接口，也没有改这份基线。这里**未在干净最新 main 构建复现**，因此不据此断言原 issue 的主线修复无效。本次只读查看 main 的 `src/backend/common/pto_ops_memory.cpp`，`tensor.view` 的 `emit_dim` 仍直接调用 `GetExprAsCode(dim)`；该文件内容 blob 为 `c025dd1926ae98c54ee949103db0307b836dffeb`。

完整 CPU 用例（无模型权重、无 NPU），保存为 `inline_dim_repro.py`，运行 `python inline_dim_repro.py --output /tmp/inline_dim_view_repro`：

```python
# SPDX-License-Identifier: Apache-2.0
"""原有tensor.view直接使用内联tensor.dim的完整CPU复现，无新增stride接口。"""
import argparse
import json
from pathlib import Path
import pypto
import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions
P = pl.dynamic("INLINE_DIM_P")
W = pl.dynamic("INLINE_DIM_W")

@pl.jit.incore
def core(source: pl.Tensor[[P, W], pl.INT8], output: pl.Out[pl.Tensor[[1, 32], pl.INT8]]):
    alias = pl.tensor.view(source, [pl.tensor.dim(source, 0), pl.tensor.dim(source, 1)])
    value = pl.load(alias, [0, 0], [1, 32])
    pl.store(value, [0, 0], output)
    return output

@pl.jit
def entry(source: pl.Tensor[[P, W], pl.INT8], output: pl.Out[pl.Tensor[[1, 32], pl.INT8]]):
    with pl.spmd(1):
        output = core(source, output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"pypto": pypto.__file__, "passed": False, "new_stride_api_used": False}
    config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"), dump_passes=True,
                            memory_planner=passes.MemoryPlanner.PYPTO)
    try:
        entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
```

本地现象：前端与 pass 执行到 codegen，记录三次 `GetExprAsCode for unsupported expression type`，随后生成类似：

```mlir
%alias = pto.make_tensor_view %arg0, shape = [, ], strides = [, %c1_index] ...
```

PTOAS 的 `expected ']'` 只是非法 MLIR 的下游症状。分配后 IR 的 TensorView 中仍可看到 `pl.tensor.dim(source, 0/1)` 的 Call 表达式。

建议核对两个层次：

1. 在 descriptor 的 shape/stride 进入 codegen 前，将常量 axis 的 `tensor.dim` 解析到对应源 TensorType 维度，或统一规范化为有定义的 INDEX Scalar，类型字段与使用方同步重写。需要覆盖静态/动态维度、内联与先绑定变量、嵌套整数算术；不要把动态维度替成样例常量。
2. `GetExprAsCode` 遇到无法表示的类型表达式时，应给出带源位置的明确错误，不要返回空字符串并继续拼出无效 MLIR。

本地临时写法是先在设备端绑定 `rows = pl.tensor.dim(source, 0)`、`width = pl.tensor.dim(source, 1)`，再传 `[rows, width]`；它不新增 Host 参数或数据回读。与 #3011 的 printer/parser round-trip、#3033 的跨 JIT metadata 丢失区分，本例直接失败在已生成 PTO 字段。这里只补充复现和修改建议，没有提交 PR 或修改标签。
