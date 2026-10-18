### Component

Other (please specify in description)

### Description

MaterializeSemanticAliases / 内存元数据传播：原地结果被重新绑定到实际目标 MemRef 后，紧随其后的 transpose_view、reshape 或 slice 仍可能保留旧分配/旧窗口。元数据 view 没有数据搬运，后续消费者却据此访问另一个未写入的窗口。这是必须别名关系丢失，不是可选的激进内存复用。

### Steps to Reproduce

在已构建的 PyPTO 环境中将下面内容保存为 `view_alias_repro.py`，执行 `python view_alias_repro.py`。这是 CPU IR/codegen 复现，不需要 NPU。

```python
# SPDX-License-Identifier: Apache-2.0
import pypto.language as pl
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.pypto_core import passes

backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

def _collect_named_tile_memrefs(program: ir.Program, names: set[str]) -> dict[str, ir.MemRef]:
    """Collect MemRefs for named tile definitions anywhere in a function body."""
    result: dict[str, ir.MemRef] = {}
    function = next(iter(program.functions.values()))

    class _MemRefCollector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):  # type: ignore[override]
            tile_type = stmt.var.type
            if stmt.var.name_hint in names and isinstance(tile_type, ir.TileType):
                assert tile_type.memref is not None
                result[stmt.var.name_hint] = tile_type.memref
            super().visit_assign_stmt(stmt)

    _MemRefCollector().visit_stmt(function.body)
    missing = names - result.keys()
    assert not missing, f"tile definitions not found: {sorted(missing)}"
    return result


@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        pool: pl.Tensor[[16, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[8, 64], pl.FP16]],
    ) -> pl.Tensor[[8, 64], pl.FP16]:
        current: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
        stale: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
        seed: pl.Tile[[16, 32], pl.FP16, pl.MemRef(current, 0, 1024), pl.Mem.Mat] = (
            pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        )
        gathered: pl.Tile[[16, 32], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.gather_row(seed, pool, [0, 0], [0, 0], [16, 32])
        )
        transposed: pl.Tile[[32, 16], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.transpose_view(gathered)
        )
        reshaped: pl.Tile[[8, 64], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.reshape(transposed, [8, 64])
        )
        return pl.tile.store(reshaped, [0, 0], out)

after = passes.materialize_semantic_aliases()(Before)
refs = _collect_named_tile_memrefs(after, {"seed", "gathered", "transposed", "reshaped"})
for name in ("gathered", "transposed", "reshaped"):
    assert ir.MemRef.same_allocation(refs[name], refs["seed"])
    assert ir.structural_equal(refs[name].byte_offset_, refs["seed"].byte_offset_)
    assert refs[name].size_ == 1024
# A second fixed pass is a no-op; this must not need a fixed-point loop.
ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)
```

### Expected Behavior

示例中 gather、transpose_view、reshape 应沿当前 seed 的分配和零偏移窗口传播；再次运行 pass 应幂等。示例主动构造pass输入的陈旧描述符，用于验证该pass已有的重绑定修复是否完整传播，不是推荐用户手写陈旧MemRef。

### Actual Behavior

最新主线的内存 pass 未修改时，`gathered` 已跟随 seed，后续 `transposed`/`reshaped` 却仍保留 stale 分配，`MemRef.same_allocation` 断言失败。类似的同base非零slice窗口会停在1024而不是应有的2048。

在最新主线对应内存实现上，新增view链测试5例中4失败、1通过（不应别名的真实transpose对照通过）。与while遍历问题一起修正后，整个 `test_memory_reuse.py` 的154项通过；旧隔离核心也能复现上面的最小脚本。

### Git Commit ID

基线 `14736f65c04a7db06547edb656f515054950555c`。最新主线负对照使用该基线的原样 `memory_reuse_pass.cpp`；核心另带独立 FloorMod codegen 修正，但本复现只执行 IR pass，不经过 codegen。

### NPU Kind

N/A (not hardware-specific)

### Host Platform

Linux (aarch64)

### Additional Context

根因位置：`src/ir/transforms/memory_reuse_pass.cpp` 的 `NormalizeIdentityCopyBuffersMutator::ReanchorInplaceOutput`。它只修注册的in-place输出；前面的RetypeApplier已经访问过body，同一轮后续view无法获知新重绑定。

建议在同一次正向遍历中同步传播已有的must-alias关系：

1. 使用注册的 `IsBufferAliasingViewOp` 区分纯view与真实数据搬运。真实transpose/extract不能因为此修正与源强行别名。
2. 让view跟随当前输入的base，并正确合成输入byte_offset与view相对offset；保留输出窗口size及slot_count/slot_index。
3. 不能只比较allocation base：同base、不同offset/size/slot同样需要更新。
4. 未经YieldFixup决定存储的if-phi，应把unsettled状态沿view/rename/in-place链传播，避免过早固定到陈旧buffer。
5. 保留幂等、非零窗口、真实transpose、独立extract、未定phi链等回归，不放宽硬件容量或独立活跃值的复用约束。

上述最小脚本的零偏移Mat链来自动态CSA整层编译中的实际现象。非零boxed Mat布局的通用物理offset模型并非本问题已证明覆盖的范围；相关IR窗口检查不能冒充所有设备布局验证。本次提供问题与建议，本地补丁保留，不提修改PR。
