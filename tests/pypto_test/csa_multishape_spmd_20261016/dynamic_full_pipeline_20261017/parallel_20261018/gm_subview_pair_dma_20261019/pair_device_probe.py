# SPDX-License-Identifier: Apache-2.0
"""独立GM子视图的设备地址验证：两行、逆序、重复、span尾界、非法索引。

只核对完整PyPTO编译链实际生成的搬运，不计性能、不替代完整CSA验证。
同址图逐轮更新device索引；BF16按INT16位型检查，覆盖NaN/Inf等所有位型。
"""
import json
import os
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
from pair_probe import pair_probe


def main():
    import torch
    import torch_npu
    import pypto
    import pypto.torch
    from pypto import passes, pypto_core
    from pypto.runtime import CompileOptions

    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, device_execution=True, performance_measured=False, checks=[],
                  pypto=pypto.__file__, core=pypto_core.__file__)
    try:
        root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
        assert root in Path(pypto_core.__file__).resolve().parents
        device_id = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(device_id)
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device("npu", device_id)
        pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
        config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        for rows in (17, 97):
            data = (torch.arange(rows * 512, dtype=torch.int32) * 1197 + 17).to(torch.int16)
            cpu = data.reshape(rows, 512).view(torch.bfloat16)
            mem = cpu.to(device)
            indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
            allocation = torch.full((4, 512), -777, dtype=torch.bfloat16, device=device)
            out = allocation[1:3]
            pointers = [x.data_ptr() for x in (mem, indices, allocation, out)]
            pair_probe(mem, indices, out, config=config)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                pair_probe(mem, indices, out, config=config)
            cases = [(0, 1), (3, 7), (7, 3), (rows - 2, rows - 1), (rows - 1, rows - 1),
                     (rows - 1, -1), (-2147483648, 2147483647), (rows, 0),
                     (rows // 3, rows * 2 // 3), (0, rows - 1), (0, 1)]
            if rows == 97:
                cases.extend([(34, 65), (35, 66), (36, 67), (66, 35)])
            for a, b in cases:
                values = torch.tensor([a, b], dtype=torch.int32)
                indices.copy_(values.to(device))
                out.fill_(float("nan"))
                graph.replay()
                torch.npu.synchronize()
                expected = torch.zeros((2, 512), dtype=torch.int16)
                for i, row in enumerate((a, b)):
                    if 0 <= row < rows:
                        expected[i] = cpu.view(torch.int16)[row]
                actual = allocation.cpu()
                assert torch.equal(actual[1:3].view(torch.int16), expected), (rows, a, b)
                assert bool((actual[[0, 3]] == torch.tensor(-777, dtype=torch.bfloat16)).all())
                assert torch.equal(mem.cpu().view(torch.int16), cpu.view(torch.int16))
                assert torch.equal(indices.cpu(), values)
                assert [x.data_ptr() for x in (mem, indices, allocation, out)] == pointers
                delta = abs(a - b)
                eligible = min(a, b) >= 0 and max(a, b) < rows and delta > 0 and min(a, b) + 2 * delta <= rows
                report["checks"].append(dict(rows=rows, indices=[a, b], pair_dma_eligible=eligible,
                                             exact_bits=True, guards=True, addresses_fixed=True))
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
