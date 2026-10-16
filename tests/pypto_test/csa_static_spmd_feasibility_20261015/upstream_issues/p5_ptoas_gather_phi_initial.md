### Component

Codegen

### Description

条件两分支均对同一个 Mat tile 执行原地 gather_row，返回的 phi 理应仍指向该存储。
连续两次这种条件更新、后接 transpose_view 和 matmul 时，PTOAS 路径会给 phi 引入新的
目标句柄并产生 Mat→Mat tmov。A2/A3 不支持这种地址空间对，PTOAS 正确拒绝编译。

触发点包含两个相连的边界：YieldFixup 在 PTOAS 模式完全跳过 if-phi 同窗口归一，
以及共享存储存在多种 view 类型时，codegen 一律拒绝复用已声明、且恰好同类型的句柄。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p5_ptoas_gather_phi.py，运行：

```bash
python -m pytest -q -o addopts='' test_p5_ptoas_gather_phi.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_conditional_gather_preserves_one_mat_buffer():
    """Two in-place branch writes must not introduce an illegal Mat->Mat copy."""

    @pl.program
    class ConditionalGather:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            base = pl.tile.create([32, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            if flag != 0:
                first = pl.tile.gather_row(base, x, [0, 0], [0, 0], [16, 64])
            else:
                first = pl.tile.gather_row(base, x, [0, 0], [16, 0], [16, 64])
            if flag != 0:
                second = pl.tile.gather_row(first, x, [16, 0], [32, 0], [16, 64])
            else:
                second = pl.tile.gather_row(first, x, [16, 0], [48, 0], [16, 64])
            transposed = pl.tile.transpose_view(second)
            rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            left = pl.tile.move(transposed, target_memory=pl.Mem.Left)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            product = pl.tile.matmul(left, right)
            output = pl.tile.store(product, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalGather)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert not any(line.count("loc=mat") == 2 for line in moves), mlir
    bases = {
        line.split("pto.subview ", 1)[1].split("[", 1)[0]
        for line in mlir.splitlines()
        if "pto.subview " in line
    }
    assert len(bases) == 1, mlir
```

### Expected Behavior

两分支原本就在同一物理窗口原地更新时，phi 继续使用该存储。后续 transpose 和 matmul 必须读到此窗口，不生成非法 Mat→Mat 拷贝，也不能只删除拷贝而让消费者读未写入的新 buffer。

### Actual Behavior

完整 CSA 在修复分支 view 问题后报错：
`'pto.tmov' op expects a supported tmov address-space pair for this target`。
附带最小回归保留连续条件 gather、transpose 和 matmul，检查没有 Mat→Mat move，
且两轮 gather 的 subview 共享同一个根句柄。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/ir/transforms/memory_reuse_pass.cpp 的 YieldFixupMutator，
以及 src/codegen/pto/pto_codegen.cpp / pto_control_flow_codegen.cpp 的共享句柄选择。
本地候选只对已同窗口的两个分支同步 phi；不同窗口不得强行合并。
多类型存储仅复用已经声明且类型完全匹配的 handle，不能修改一个 SSA handle 的类型。

验证边界：

附带最小 CPU 回归在本地候选上通过，完整 CSA 已越过该 Mat→Mat 错误。
本问题属于 PTOAS 备用路径的正确性 bug，不是当前 PYPTO 路径的性能优化。
尚未用完整 CSA 真机验证该 PTOAS 分支，不声明完整算子通过。

Related: #1956、#2693；#2902 是 AutoTileMatmulL0 对已有 Mat 操作数额外搬运，本例是条件 phi / 多类型共享句柄，根因不同。
