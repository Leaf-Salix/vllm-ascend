### Component

Codegen / 算子类型推导（tile.gather_compare）

### Description

A3 上 `pl.tile.gather_compare` 的计数输出由编译器内部创建为 `[1, src_rows] INT32`。
当输入只有一行（Top-K 阈值压缩很常见）时，生成物理 `[1,1]` Tile，行宽仅 4B，
被 PTOAS 的 32B UB 对齐检查拒绝。调用方不能通过预分配 `counts[1,8]` 修复内部输出。

这不是要求逻辑计数增加到 8；应保留一个有效计数、使用合法物理容量。

### Steps to Reproduce

以下脚本仅在 CPU 生成 A3 代码，不需要 NPU、模型权重或 vLLM。
配置好 PyPTO/PTOAS 与受支持 Simpler 后运行：

```python
import pypto.language as pl


@pl.jit.incore
def compare_kernel(
    src: pl.Tensor[[1, 64], pl.FP32],
    selected: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
    counts: pl.Out[pl.Tensor[[1, 8], pl.INT32]],
):
    values = pl.load(src, [0, 0], [1, 64])
    tmp = pl.tile.create([1, 512], dtype=pl.UINT8)
    ids, count = pl.tile.gather_compare(values, pl.cast(0.5, pl.FP32), tmp,
                                       cmp_mode="gt", out_cols=64)
    pl.store(ids, [0, 0], selected)
    pl.store(count, [0, 0], counts)
    return selected, counts


@pl.jit
def entry(src: pl.Tensor[[1, 64], pl.FP32],
          selected: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
          counts: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
    with pl.spmd(1):
        selected, counts = compare_kernel(src, selected, counts)
    return selected, counts


from pypto import passes
from pypto.runtime import CompileOptions
config = CompileOptions(platform="a2a3", output_dir="/tmp/gather-count-repro",
                        dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
```

### Expected Behavior

内部计数 Tile 的物理列数对齐到 8 个 INT32，valid_shape 仍是 `[1, src_rows]`。
调用方只读取/写出真实有效计数。至少覆盖 src_rows=1,2,7,8,9,16。

### Actual Behavior

```text
pto.alloc_tile expects result row-major none_box tile row byte size
(cols * sizeof(dtype)) to be 32-byte aligned, but got 4 bytes
```

生成的计数结果为：

```mlir
!pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=1, v_row=?, v_col=?,
              blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

### 定位与修改建议

`src/ir/op/tile_ops/gather.cpp::DeduceTileGatherCompareType` 当前同时设置：

```cpp
cdst_view.valid_shape = {one_expr, rows_expr};
std::vector<ExprPtr> cdst_shape = {one_expr, rows_expr};
```

建议将计数的物理最后一维按 32B 对齐，保留真实 valid_shape；若需要把逻辑 shape
保持原样，则在统一 Tile 物理布局合法化阶段处理。不能仅关掉 PTOAS 检查。
还需回归 tuple 输出、tile.read(count)、store 的有效范围，以及非 8 倍数行数。

### Git Commit ID

复现 PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`，PTOAS
`e5c1c432af8249bac2f905c7648ebafefec3c20a`。2026-10-20 只读核对 GitHub main
的相同函数仍使用上述物理形状；没有宣称完成最新 main 的整仓重建。

### NPU Kind / Host Platform

Ascend A3（Ascend910_9392），Linux aarch64，Python 3.10，CANN 9.2.0-beta.2。
失败发生在 CPU lowering，未运行设备。

### Additional Context

八行输入避开此对齐问题后，另有 PTOAS compare-form TGATHER 与当前 ISA ABI
不一致的独立 C++ 编译问题，会单独报告；修复本 issue 不代表整条 gather_compare
已经能上卡。本 issue 不附 PR、不设置标签。
