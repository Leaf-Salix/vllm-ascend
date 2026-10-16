### Component

Codegen

### Description

在 memory_planner=PTOAS / emit_tile_addr=False 下，分支内对列向量作
reshape→add→reshape，分支外再对 phi 结果 reshape。当前 view no-op 判断只比较
预声明目标句柄的类型与结果类型。类型相同并不说明旧 MemRef 代表句柄就是当前 phi。
分支外消费可能错误地选到只在分支内定义的 producer，导致非法 SSA dominance。
transpose_view 使用同类 no-op 判断，也需要同一约束。

这是生成 IR 的合法性问题；PTOAS 对该 IR 的拒绝是合理的，不要求 PTOAS 放宽验证。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p4_ptoas_branch_view.py，运行：

```bash
python -m pytest -q -o addopts='' test_p4_ptoas_branch_view.py
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

def test_ptoas_reshape_reads_phi_handle_after_branch():
    """A post-if view cannot reuse a same-typed but branch-local old handle."""

    @pl.program
    class BranchView:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 1], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            base = pl.load(x, [0, 0], [32, 1])
            if flag != 0:
                row = pl.reshape(base, [1, 32])
                updated = pl.add(row, 1.0)
                column = pl.reshape(updated, [32, 1])
                result = column
            else:
                result = base
            after = pl.reshape(result, [1, 32])
            head = pl.tile.extract(after, 0, 0, [1, 16], target_memory=pl.Mem.Vec)
            output = pl.store(head, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, BranchView)
    mlir = _codegen(optimized, emit_tile_addr=False)
    extract = next(line for line in mlir.splitlines() if "pto.textract ins(" in line)
    source = extract.split("ins(", 1)[1].split(",", 1)[0]
    view = next((line.strip() for line in mlir.splitlines() if line.strip().startswith(source + " =")), "")
    assert "= pto.treshape " in view, mlir
    phi = view.split("= pto.treshape ", 1)[1].split()[0]
    assert any("pto.tmov" in line and "outs(" + phi + " :" in line for line in mlir.splitlines()), mlir
```

### Expected Behavior

分支后的 view 必须从实际 phi/源句柄派生。没有固化物理地址时，不能仅凭同类型省略 view。后续 extract 应读取由 phi 派生的 treshape，其源定义支配使用点。

### Actual Behavior

真实 CSA 的 PTOAS 路径曾产生两处分支外引用分支内旧句柄的问题。最小回归检查 extract 的源确实由 phi 的 treshape 产生，而不是只检查能够打印 MLIR。

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

src/backend/common/pto_ops_datamove.cpp 的 tile.reshape / tile.transpose_view。
本地候选仅在固定地址模式或实际 source handle 与已绑定 target 相同时执行原 no-op，
否则从实际 source 发出 view。此处不修改内存容量、算术、Tile 或流水深度。

验证边界：

附带最小 CPU 回归在本地候选上通过，原完整 CSA 的两处 SSA 合法性错误也已越过。
本问题来自 PTOAS 备用编译路径；当前完整 CSA 默认 PYPTO 路径不依赖此项。
完整 PTOAS 路径仍存在独立的 Mat 容量问题，不能将本项修复记为整层编译或真机通过。

Related: #1642（固定地址 view 丢失依赖链，本例为无地址模式下分支后的 SSA 句柄）、#2693。
