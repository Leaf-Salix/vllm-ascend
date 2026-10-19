# SPDX-License-Identifier: Apache-2.0
"""GM→L1 三槽的真机数据验证；不测性能，不替代完整 CSA/跨核验证。"""
import json
import os
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
from triple_slot_probe import triple_slot


def main():
    import torch
    import torch_npu
    import pypto
    import pypto.torch
    from pypto import passes, pypto_core
    from pypto.runtime import CompileOptions

    outdir = Path(sys.argv[1]).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, device_execution=False, performance_measured=False,
                  full_csa_validated=False, checks=[], core=pypto_core.__file__)
    try:
        root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
        assert root in Path(pypto_core.__file__).resolve().parents
        did = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(did)
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device("npu", did)
        pypto.torch.init(device=did, platform="a2a3", runtime="tensormap_and_ringbuffer")
        config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        columns = torch.tensor([0, 31, 32, 63, 64, 127, 128, 255,
                                256, 383, 384, 447, 448, 479, 480, 511])
        query_cpu = torch.zeros((16, 512), dtype=torch.bfloat16)
        query_cpu[torch.arange(16), columns] = 1
        query = query_cpu.to(device)
        for rows, steps in ((193, 5), (257, 8)):
            table_cpu = ((torch.arange(rows).view(-1, 1) * 13 + torch.arange(512) * 3) % 127 - 63).to(torch.bfloat16)
            table = table_cpu.to(device)
            base_indices = (torch.arange(steps * 128, dtype=torch.int32) * 19) % rows
            base_indices[::11] = -1
            base_indices[3::29] = rows
            base_indices[5::31] = 2147483647
            base_indices[7::37] = -2147483648
            indices = base_indices.reshape(-1, 1).to(device)
            allocation = torch.full((steps * 16 + 2, 128), -777.0, dtype=torch.float32, device=device)
            output = allocation[1:-1]
            pointers = [x.data_ptr() for x in (table, indices, query, output)]
            triple_slot(table, indices, query, output, config=config)
            torch.npu.synchronize()
            report["device_execution"] = True
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                triple_slot(table, indices, query, output, config=config)
            for variant, values in (("A", base_indices), ("B", base_indices.flip(0)), ("A", base_indices)):
                indices.copy_(values.reshape(-1, 1).to(device))
                output.fill_(float("nan"))
                graph.replay()
                torch.npu.synchronize()
                valid = (values >= 0) & (values < rows)
                gathered = torch.zeros((steps * 128, 512), dtype=torch.bfloat16)
                gathered[valid] = table_cpu[values[valid].long()]
                # 单位基query使每个结果为一个精确BF16→FP32值，独立核对ND2NZ
                # 全宽关键列与三槽周转，不借用被测mgather作为oracle。
                expected = gathered[:, columns].reshape(steps, 128, 16).transpose(1, 2).reshape(steps * 16, 128).float()
                got = allocation.cpu()
                assert torch.equal(got[1:-1], expected), (rows, steps, variant)
                assert bool((got[[0, -1]] == -777).all())
                assert torch.equal(table.cpu(), table_cpu)
                assert torch.equal(indices.cpu(), values.reshape(-1, 1))
                assert torch.equal(query.cpu(), query_cpu)
                assert pointers == [x.data_ptr() for x in (table, indices, query, output)]
                report["checks"].append(dict(rows=rows, steps=steps, variant=variant,
                                             exact=True, guards=True, same_addresses=True))
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (outdir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
