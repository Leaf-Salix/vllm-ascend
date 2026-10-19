## 需求与现状

希望在同一个 InCore 内，根据 device 上的地址/跨度建立**零拷贝 GM 子视图**，再显式 `pl.load`。用途是将两条任意、合法、升序的 BF16 行合成一次带步长的 GM→UB 搬运，减少稀疏 Attention 的逐行小 DMA。

这是当前接口明确不支持的组合/能力扩展，**不是已经证明的错误输出 bug，也不是现有 CSA 正确运行的必需修复**。目前根 GM Tensor 的 `tensor.view` 可以编译；`tensor.slice` 的结果在 ConvertTensorToTileOps 先变成 Tile，随后 `tensor.view` 显式拒绝。没有绕过检查，没有修改共享安装，也没有真机性能结论。

相关：#1964 讨论 GM reinterpret；#2087 是 orchestration→incore 的动态 stride 错误。这里全部发生在同一 InCore 内，表现为显式拒绝，区别于跨边界 stride 静默错误。

## 最小复现与成功对照

本地独立 PyPTO worktree 基点 `d01de8523`（含 CSA 项目已记录的本地编译器修改），A3/a2a3，CPU-only，PYPTO memory planner，无样例 shape、无 NPU 初始化。当前 main `2964333d45a2b861b78977101278099c8812fc8b` 的同一拒绝逻辑也已只读核对：[src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp](https://github.com/hw-native-sys/pypto/blob/2964333d45a2b861b78977101278099c8812fc8b/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp)。**未把这次只读核对宣称成 main 构建复现。**

以下代码保存为 `minimal_repro.py`：

```python
# SPDX-License-Identifier: Apache-2.0
"""只比较根GM视图与GM切片后视图的编译行为，不声明两者读取相同行。"""
import pypto.language as pl

ROWS = pl.dynamic("VIEW_SOURCE_ROWS")


@pl.jit.incore
def slice_view_core(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    # 要求调用者提供至少4行。元素数4*512 == 2*1024，视图没有扩大存储。
    span = pl.tensor.slice(x, [4, 512], [0, 0])
    pitched = pl.tensor.view(span, shape=[2, 1024])
    tile = pl.load(pitched, [0, 0], [2, 512])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit.incore
def root_view_core(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    # 对照仅确认直接根GM Tensor可view；读取0/1行，不是上述0/2行的数值对照。
    rows = pl.tensor.dim(x, 0)
    direct = pl.tensor.view(x, shape=[rows, 512])
    tile = pl.load(direct, [0, 0], [2, 512])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit
def slice_view(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    with pl.spmd(1, name_hint="slice_view"):
        out = slice_view_core(x, out)
    return out


@pl.jit
def root_view(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    with pl.spmd(1, name_hint="root_view"):
        out = root_view_core(x, out)
    return out

```

安装并构建正常 PyPTO 环境、配置 PTOAS 后，运行：

```python
from pathlib import Path
from pypto import passes
from pypto.runtime import CompileOptions
from minimal_repro import root_view, slice_view

for name, entry in (("root", root_view), ("slice", slice_view)):
    try:
        entry._resolve_kernel_artifact(
            (),
            {"config": CompileOptions(
                platform="a2a3",
                output_dir=str(Path("view_repro") / name),
                dump_passes=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
            )},
            allow_signature_mode=True,
        )
        print(name, "compiled")
    except Exception as exc:
        print(name, type(exc).__name__, str(exc))
```

实测：

| 用例 | CPU 编译结果 | 说明 |
|---|---|---|
| 根 GM → `view` → `load` | 完整生成 PTO/C++ | 证明 GM view 本身可用 |
| GM → `slice([4,512])` → `view([2,1024])` → `load([2,512])` | ConvertTensorToTileOps 拒绝 | 两次描述的元素数同为 2048 |

失败信息的核心部分：

```text
tensor.view in an InCore function requires a GM Tensor input that remains tensor-like through ConvertTensorToTileOps; viewing the result of an op lowered to Tile is not supported
```

根视图对照读取第 0/1 行；切片视图想读取第 0/2 行。这里只比较编译能力，**两者不是数值等价对照**。调用切片示例需要原 allocation 至少 4 行；没有扩张存储、伪造 shape 或读取片外数据。

## 动态场景为什么需要这个能力

真实场景中，行号 `a,b` 在 device Tensor。令 `lo=min(a,b)`、`delta=abs(b-a)`，先证明两行有效、`delta>0`，并证明完整 `2*delta` 行 span 从 `lo` 起仍在 allocation 内，才考虑：

```python
span = pl.tensor.slice(cache, [2 * delta, 512], [lo, 0])
pitched = pl.tensor.view(span, shape=[2, delta * 512])
pair = pl.load(pitched, [0, 0], [2, 512])
```

这段形状乘积严格守恒，只需要加载两行，不应先把整个大 span 搬进 UB。A3 PTO-ISA GM→UB 的 `TLoadInstrGm2ubNative` 有 32 位 byte gap；两段各 1024B、间隔 `(delta-1)*1024`，但还必须检查该编码上界。降序在 UB 恢复；重复行、非法行、span 不完整或 gap 太大退回原逐行实现。硬件接口可表达不等于已测得加速。

## 修改建议（尚未实现，不能直接删除当前保护检查）

可选方案 A：明确提供 GM subview/strided-load API，支持 device Scalar 的 offset 与正 stride，并规定其不物化 Tile；保留底层 base、offset、shape、strides、valid_shape 与 alias/lifetime 信息。这样不会改变已有 `tensor.slice` 在 InCore 中表示 Tile 提取/隐式加载的语义。

可选方案 B：仅对可证明是 descriptor 链的 `GM slice → view → explicit load` 保留 GM 表示。建议在 Tensor→Tile 之前分析消费者，而非在错误处把 Tile 强制转回 Tensor：

```text
if source_is_GM and all_uses_are_supported_alias_or_explicit_load:
    preserve_subview_as_GM_descriptor(base, offset, shape, strides, validity)
else:
    keep_existing_tensor_to_tile_behavior()
```

两方案都应保持：

- 非连续列切片不能按 dense shape 猜 stride；原 memref/别名和有效范围必须完整传递。
- 真正计算得到的 Tile 不能重新解释成 GM Tensor，现有拒绝仍须保留。
- 动态偏移/跨度不能通过 Host 回读求值；目标形状、加载范围、编码宽度与整数溢出需检查。
- 验证 slice→view→load 不产生大 span 的隐式 UB 分配/搬运；生成 `subview/make_tensor_view` 的地址与 stride 正确。
- 增加直接根 view、连续子视图、动态 pitch、列切片、分支/循环和真实 Tile 拒绝等回归；设备验证应包含降序、重复、尾边界及 fallback。

如果已有另一种正式接口能直接表达上述 GM offset+stride，请指出对应写法，可直接替代本建议。当前仅提交 issue 与实现方向，不提交 PR。
