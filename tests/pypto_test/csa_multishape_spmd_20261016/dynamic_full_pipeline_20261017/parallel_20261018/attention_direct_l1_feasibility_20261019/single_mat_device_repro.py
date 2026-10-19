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
    ordered = "--ordered" in sys.argv
    if ordered:
        from private_isa_scope import install
        install(folder / "isa_audit")
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
