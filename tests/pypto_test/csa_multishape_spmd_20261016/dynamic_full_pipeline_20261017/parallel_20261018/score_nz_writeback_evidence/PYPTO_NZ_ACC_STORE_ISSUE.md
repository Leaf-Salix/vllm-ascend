### 主旨

扩展现有 `pl.NZ`：支持 AIC 的 FP32 Acc Tile 直接写入 NZ 格式的 GM Tensor。
这是 #2533 只读首期能力的后续需求，不把现有明确限制误报为算错的 bug。
建议复用现有 `pl.store` / `tile.store` 和 `TensorLayout::NZ`，无需另造一个布局枚举。

### 动机与实际场景

正在优化 A3 上完整的动态 CSA 单 SPMD。Indexer Score 的转置数据流能降低
Mat→L0搬运，但此前窄列 ND 写回增加了 FIX 成本。希望评估同一份 FP32 Acc
经 NZ 写回，供后继 AIV 读取；数学、归约和核间 READY/ACK 都保持原样。

已经在完整动态 CSA 的生成代码中做了临时硬件实验：只将16处完整 `64×16`
FP32 Acc store 的目标描述符改为每panel独立 NZ/C0=8，并让AIV配套解读字节布局。
128K/B16/S6、正式权重的 A→B→A 三次输出、Top-K/score、cache/state 和保护区
已精确通过。**当前完整性能对照尚在进行，没有宣称性能收益；生成代码替换仅是
硬件能力实验，不是可交付的 PyPTO 接口。** 后续还需正式API和对应测试。

### 当前主线及可复现限制

已只读核对主线 `14736f65c04a7db06547edb656f515054950555c`：
[BlockNzTensorViews 的只读检查](https://github.com/hw-native-sys/pypto/blob/14736f65c04a7db06547edb656f515054950555c/src/ir/transforms/block_nz_tensor_views_pass.cpp#L447) 仍无条件拒绝 `tile.store` 的 NZ 目标。
本地独立worktree配套核心库运行下面的编译pass复现，失败点同样是
`block_nz_tensor_views`，没有运行NPU。

错误摘要：

```text
NZ layout is read-only: an NZ tensor cannot be a store destination.
```

最小复现（在已构建PyPTO环境运行）：

```python
import pypto.language as pl
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.pypto_core import passes

@pl.program
class StoreAccumulatorToNz:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[64, 64], pl.FP16],
             w: pl.Tensor[[64, 16], pl.FP16],
             out: pl.Tensor[[64, 16], pl.FP32, pl.NZ]):
        lhs = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        rhs = pl.load(w, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
        pl.store(acc, [0, 0], out)
        return out

reset_for_testing()
set_backend_type(BackendType.Ascend910B)
program = StoreAccumulatorToNz
for name in (
    "inline_functions", "unroll_loops", "ctrl_flow_transform", "convert_to_ssa",
    "simplify", "normalize_stmt_structure", "flatten_call_expr",
    "outline_hierarchy_scopes", "outline_incore_scopes", "outline_cluster_scopes",
    "convert_tensor_to_tile_ops", "optimize_orch_tensors", "lower_composite_ops",
    "flatten_tile_nd_to_2d", "block_nz_tensor_views",
):
    program = getattr(passes, name)()(program)
```

### 期望接口与生成结果

上面的 `pl.store(acc, [0, 0], out)` 应直接生成 NZ 目标描述符，并保持逻辑二维
Tile及数学不变。对于FP32的 `64×16`，PTO-native C0=8：

```cpp
using NzShape = pto::Shape<1, 2, 4, 16, 8>;
using NzStride = pto::Stride<1024, 512, 128, 8, 1>;
pto::GlobalTensor<float, NzShape, NzStride, pto::Layout::NZ> dst(
    output_address, NzShape(), NzStride());
pto::TSTORE(dst, accumulator);
```

物理顺序为 `[N/8, M/16, 16, 8]`。同一panel仍占 `64×16×4` 字节；这不是
改变数学结果，只是明确写回的存储布局。

[A2/A3 的 TStoreAccNz2nz](https://github.com/hw-native-sys/pto-isa/blob/327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d/include/pto/npu/a2a3/TStore.hpp)
已经有该硬件路径，FP32的C0=8会启用channel split。

### 修改建议

1. `BlockNzMutator` 仍检查全部操作数，但将合法 `tile.store` 的目标（参数2）
   单独派发到 NZ store 重写；不要简单删除当前保护。
2. 复用现有逻辑NZ→blocked Tensor类型改写，以及偏移对齐/非负证明。
   把store的逻辑offset和partition shape改为对应blocked坐标，保持Acc的二维
   Tile类型和有效行列，保持返回Tensor别名正确。
3. 当前后端可以表达NZ `GlobalTensor`。确认store emitter使用blocked目标和
   `layout=nz`，不重新推断成ND；不要引入额外GM转换kernel。
4. 复制并校验已有store属性（包括unit-flag阶段等），对未支持的组合明确拒绝。
   首期可只支持A2/A3、FP32 Acc→FP32 GM、完整且已证明对齐的分区；Vec store、
   部分fractal、不能证明的符号偏移、原子及dtype转换不应随手放开。
5. 注意PTO-ISA对FP32 NZ的C0允许8/16，但INT32写回另有C0=16限制；不能把
   FP32路径机械推广到所有dtype。布局规则以声明的GM字节格式为准。

建议控制流示意（伪代码，尚未作为编译器补丁验证）：

```cpp
if (IsOp(op, "tile.store") && IsNzTensor(op->args_[2])) {
  CheckSupportedAccNzStore(op);       // memory space / dtype / alignment / attrs
  auto blocked_offsets = BlockNzOffsets(op->args_[1], facts);
  auto blocked_window = BlockNzWindow(GetSourceValidShape(op), output_dtype);
  return RebuildNzStore(op, blocked_offsets, blocked_window); // preserve attrs/alias
}
// Keep rejecting all other unsupported NZ consumers/writers.
```

### 建议验收

- 独立pass测试：参数2目标重写、返回别名、属性保存、行/列偏移及对齐边界。
- 继续拒绝不支持的Vec/原子/部分fractal/dtype组合，不能移除保护后静默误写。
- A3设备测试：小整数输入精确matmul，按NZ物理顺序解包，与ND参考逐bit比较。
- 非零行/列偏移、多个不重叠panel及多核独占写区，检查完整allocation与保护区。
- 完整动态CSA仍需用正式API复验，临时生成代码实验不能替代该验收。

### 替代方案与边界

保持ND写回仍然正确，但无法以受支持的DSL接口评估NZ数据流。额外布局转换会
增加搬运与调度；长期修改生成代码也不适合作为正式实现。
本需求是编译器能力扩展；最终算子是否更快仍以完整同轮L1为准。
