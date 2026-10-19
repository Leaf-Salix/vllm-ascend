### 问题与定位

A3 上 `MGatherGm2L1RowImpl< GatherOOB::Zero >` 先用 `pto_create_cbuf_matrix` 清零整个 L1 Tile，随后立即对相同目标逐行发出 ND2NZ DMA。当前实现没有在清零与行拷贝之间建立 MTE2 顺序约束。

单核、单个 Mat Tile、一次 MGATHER 的真机复现中，少量应为非零的结果不稳定地变成 0。仅在清零后增加 `pipe_barrier(PIPE_MTE2)`，相同输入的 12 次调用全部精确通过；动态三槽探针的 6 项同址 A→B→A 图重放也通过。

这与 #178 不同：#178 是 GM→UB 的索引 DMA 后 Scalar 读取，本例是 **GM→L1 的 ZERO 初始化与后续 ND2NZ 对同一 L1 存储的写顺序**。

### 环境和范围

- Ascend A3，CANN 9.2.0-beta.2。
- 实际编译使用的 PTO-ISA commit：`327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
- `include/pto/npu/a2a3/MGather.hpp` 的 `MGatherGm2L1RowImpl`，Row 模式、BF16、`GatherOOB::Zero`、目标 Mat/NZ 128×512。
- 最小复现不含 CSA、不含 GM 子视图、不含动态循环、不含核间同步，也不依赖模型权重。
- PyPTO 用来生成一次 `MGATHER<Coalesce::Row, GatherOOB::Zero>` 和后续 QK；生成源码有 MTE2→MTE1 获取。外部同步不能把 MGATHER 函数内部两种写操作分开排序。
- 建议补丁只作用于私有头文件副本，编译路径有审计；没有改共享安装或推送组件分支。

### 真机观察

table 为193×512 BF16，idx 为1×128 INT32，包含重复/负数/越界项；query为16×512单位基，覆盖列0、31、32、63、64、127、128、255、256、383、384、447、448、479、480、511。
这样QK输出是被选BF16数据的精确FP32值，没有累加误差借口，也能观察完整512列中的关键布局边界。

同样的12组输入：

| 版本 | 每次调用的差异个数 |
|---|---|
| 原实现 | 0, 0, 1, 4, 3, 2, 0, 0, 0, 0, 0, 0 |
| 仅补下方MTE2屏障 | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |

先前动态三槽未加屏障时有78个元素差异，eager与replay也不同；补屏障后5/8步两个规格、各A→B→A的6项输出/保护/同地址检查全部通过。这是原语级证据，不宣称完整CSA或所有dtype都已验证。

### 建议修改

```diff
--- a/include/pto/npu/a2a3/MGather.hpp
+++ b/include/pto/npu/a2a3/MGather.hpp
@@ -307,6 +307,8 @@
         constexpr uint32_t kColBlocks = kTileCols / kC0;
         int64_t repeatConfig = (static_cast<int64_t>(kTileRows) << 16) | static_cast<int64_t>(kColBlocks);
         pto_create_cbuf_matrix((__cbuf__ uint16_t*)dstPtr, repeatConfig, 0);
+        // ZERO initialization and ND2NZ write overlapping L1 rows; complete fill first.
+        pipe_barrier(PIPE_MTE2);
     }
 
     for (uint32_t r = 0; r < validRow; r++) {
```

请维护者核对该架构的具体指令顺序合同；当前对照支持缺少内部写顺序约束这一定位。不能只在整个 MGATHER 返回之后加屏障来阻止此前已经发生的写覆盖。

### 完整最小复现

在已有 PyPTO、torch_npu、Simpler 的 A3 测试环境中保存下列脚本，显式选择空闲卡运行（本次所有测试均通过本机任务队列）：

```bash
TASK_DEVICE=0 python single_mat_public_repro.py /tmp/mgather_zero_original
```

第一次没有差异不代表不存在，因为错误有时序敏感性；脚本会记录全部12次调用，而不是首个失败就退出。补丁版应使用修改后的独立PTO-ISA头文件根编译同一个脚本，不能复用旧二进制。

```python
# SPDX-License-Identifier: Apache-2.0
"""单个 GM→Mat MGATHER ZERO 的独立真机复现；不含 GM 子视图、三槽或 CSA。"""
import json
import os
import sys
import traceback
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def single_mat_core(
    table: pl.Tensor[[193, 512], pl.BF16],
    indices: pl.Tensor[[1, 128], pl.INT32],
    query: pl.Tensor[[16, 512], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
):
    q = pl.load(query, [0, 0], [16, 512], target_memory=pl.Mem.Mat)
    rows = pl.tile.mgather(table, indices, coalesce="row", gather_oob="zero", target_memory=pl.Mem.Mat)
    score = pl.matmul(q, pl.tile.transpose_view(rows), out_dtype=pl.FP32)
    pl.store(score, [0, 0], output)
    return output


@pl.jit
def single_mat(
    table: pl.Tensor[[193, 512], pl.BF16],
    indices: pl.Tensor[[1, 128], pl.INT32],
    query: pl.Tensor[[16, 512], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
):
    with pl.spmd(1):
        output = single_mat_core(table, indices, query, output)
    return output


def main():
    import torch
    import torch_npu
    import pypto
    import pypto.torch
    from pypto import passes
    from pypto.runtime import CompileOptions

    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    ordered = False
    result = dict(passed=False, ordered=ordered, checks=[], device_execution=False,
                  performance_measured=False, scope="one Mat MGATHER, no loop or GM subview")
    try:
        did = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(did)
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device("npu", did)
        pypto.torch.init(device=did, platform="a2a3", runtime="tensormap_and_ringbuffer")
        config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        columns = torch.tensor([0,31,32,63,64,127,128,255,256,383,384,447,448,479,480,511])
        table_cpu = ((torch.arange(193).view(-1,1)*13 + torch.arange(512)*3) % 127 - 63).to(torch.bfloat16)
        query_cpu = torch.zeros((16,512), dtype=torch.bfloat16)
        query_cpu[torch.arange(16), columns] = 1
        table, query = table_cpu.to(device), query_cpu.to(device)
        indices = torch.zeros((1,128), dtype=torch.int32, device=device)
        allocation = torch.full((18,128), -777.0, device=device)
        output = allocation[1:17]
        for variant in range(12):
            values = (torch.arange(128, dtype=torch.int32)*19 + variant*17) % 193
            values[::11] = -1
            values[3::29] = 193
            values[5::31] = 2147483647
            indices.copy_(values.reshape(1,128).to(device))
            output.fill_(float("nan"))
            single_mat(table, indices, query, output, config=config)
            torch.npu.synchronize()
            result["device_execution"] = True
            valid = (values >= 0) & (values < 193)
            selected = torch.zeros((128,512), dtype=torch.bfloat16)
            selected[valid] = table_cpu[values[valid].long()]
            expected = selected[:,columns].T.float()
            actual = output.cpu()
            different = (actual != expected).nonzero()
            result["checks"].append(dict(variant=variant, mismatches=len(different),
                first=[dict(row=i,col=j,actual=float(actual[i,j]),expected=float(expected[i,j]))
                       for i,j in different[:8].tolist()]))
            assert bool((allocation.cpu()[[0,-1]] == -777).all())
            assert torch.equal(table.cpu(), table_cpu)
            assert torch.equal(query.cpu(), query_cpu)
            assert torch.equal(indices.cpu(), values.reshape(1,128))
        result["passed"] = all(c["mismatches"] == 0 for c in result["checks"])
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    (folder / "result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
