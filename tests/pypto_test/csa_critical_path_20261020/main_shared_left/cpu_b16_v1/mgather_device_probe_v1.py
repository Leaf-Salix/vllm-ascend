# SPDX-License-Identifier: Apache-2.0
"""同址图中核对现有A3 MGATHER的物理行、负哨兵、上界及BF16逐bit拷贝。

只验证局部搬运合同，不计性能，不代替完整CSA同步/精度验证。输入行数动态，
固定64×512是完整Attention当前采用的搬运tile，设备索引在A→B→C→A间变化。
"""
import json
import os
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
import pypto.language as pl

ROWS = pl.dynamic("MGATHER_PROBE_ROWS")


@pl.jit.incore
def gather_core(
    mem: pl.Tensor[[ROWS, 512], pl.BF16],
    indices: pl.Tensor[[1, 64], pl.INT32],
    out: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
):
    rows = pl.load(indices, [0, 0], [1, 64], target_memory=pl.MemorySpace.Vec)
    value = pl.tile.mgather(mem, rows, coalesce="row", gather_oob="zero")
    out = pl.store(value, [0, 0], out)
    return out


@pl.jit
def gather_probe(
    mem: pl.Tensor[[ROWS, 512], pl.BF16],
    indices: pl.Tensor[[1, 64], pl.INT32],
    out: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
):
    with pl.spmd(1, name_hint="physical_row_mgather_probe"):
        out = gather_core(mem, indices, out)
    return out


def main():
    import torch
    import torch_npu
    import pypto.torch
    from pypto import passes
    from pypto.runtime import CompileOptions

    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, device_execution=True, performance_measured=False, checks=[])
    try:
        torch.npu.set_device(int(os.environ["TASK_DEVICE"]))
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device("npu", int(os.environ["TASK_DEVICE"]))
        config = CompileOptions(platform="a2a3", output_dir=str(folder / "build"),
                                dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
        for row_count in (17, 97):
            # 覆盖全部bit模式（含NaN/Inf）；只做搬运，比较INT16位型。
            bits = (torch.arange(row_count * 512, dtype=torch.int32) * 1197 + 17).to(torch.int16)
            mem_cpu = bits.reshape(row_count, 512).view(torch.bfloat16)
            a = ((torch.arange(64) * 37 + row_count - 1) % row_count).to(torch.int32).reshape(1, 64)
            b = a.clone()
            b[0, ::3], b[0, 1::7], b[0, 2::11], b[0, 5::13] = -1, row_count, -2147483648, 2147483647
            c = torch.full_like(a, -1)
            mem, indices = mem_cpu.to(device), a.to(device)
            allocation = torch.full((66, 512), -777, dtype=torch.bfloat16, device=device)
            out = allocation[1:65]
            addresses = [x.data_ptr() for x in (mem, indices, allocation, out)]
            gather_probe(mem, indices, out, config=config)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                gather_probe(mem, indices, out, config=config)
            for name, values in (("A", a), ("B_mixed", b), ("C_all_invalid", c), ("A_again", a)):
                indices.copy_(values.to(device))
                out.fill_(float("nan"))
                graph.replay()
                torch.npu.synchronize()
                expected = torch.zeros((64, 512), dtype=torch.int16)
                valid = (values.flatten() >= 0) & (values.flatten() < row_count)
                expected[valid] = mem_cpu.view(torch.int16)[values.flatten()[valid].long()]
                actual = allocation.cpu()
                assert torch.equal(actual[1:65].view(torch.int16), expected), (row_count, name)
                assert bool((actual[[0, 65]] == torch.tensor(-777, dtype=torch.bfloat16)).all())
                assert [x.data_ptr() for x in (mem, indices, allocation, out)] == addresses
                assert torch.equal(mem.cpu().view(torch.int16), mem_cpu.view(torch.int16))
                assert torch.equal(indices.cpu(), values)
                report["checks"].append(dict(rows=row_count, case=name, exact_bits=True, guards=True,
                                             addresses_fixed=True, inputs_preserved=True))
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
