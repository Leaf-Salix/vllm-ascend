# SPDX-License-Identifier: Apache-2.0
"""真实A3的L1 gather哨兵用例；无Native、无性能采样、无共享依赖修改。"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import pypto.torch
from pypto import passes
from pypto.runtime import CompileOptions

from qk_direct_l1_probe_entry import entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    # BF16可精确表示的小整数，每行和每列都有变化；单位阵乘不引入舍入差。
    src_cpu = ((torch.arange(768)[:, None] * 7 + torch.arange(512)[None, :] * 3)
               .remainder(251) - 125).to(torch.bfloat16)
    indexes = ((torch.arange(128) * 37 + 127) % 384).to(torch.int32)
    src, idx = src_cpu.to(device), indexes.to(device)
    select = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=device)
    eye = torch.eye(128, dtype=torch.bfloat16, device=device)
    allocation = torch.full((386, 512), -777.0, device=device)
    out = allocation[1:385]
    options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
    report = {"passed": False, "checks": [], "scope": "GM到三槽L1搬运及邻接哨兵，不是完整CSA"}
    try:
        entry(src, idx, select, eye, out, config=options)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            entry(src, idx, select, eye, out, config=options)
        cases = [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0),
                 (1, 1, 1, 1), (1, 1, 31, 32), (2, 1, 127, 1), (0, 0, 0, 0)]
        for case in cases:
            select.copy_(torch.tensor(case, dtype=torch.int32, device=device))
            out.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            expected = src_cpu[384:].float().clone()
            slot, mode, start, count = case
            if mode == 0:
                expected[slot * 128:(slot + 1) * 128] = src_cpu[indexes.long()].float()
            else:
                begin = slot * 128 + start
                expected[begin:begin + count] = src_cpu[:count].float()
            actual = allocation.cpu()
            different = int((actual[1:385] != expected).sum())
            guard = bool((actual[[0, 385]] == -777).all())
            report["checks"].append({"selector": case, "different_elements": different, "guard": guard})
            if different or not guard:
                bad_rows = (actual[1:385] != expected).any(dim=1).nonzero().flatten().tolist()
                report["bad_rows"] = bad_rows
                raise AssertionError(f"L1搬运哨兵失败：{case}, different={different}, guard={guard}")
        assert torch.equal(src.cpu(), src_cpu) and torch.equal(idx.cpu(), indexes)
        report.update(passed=True, same_graph=True, read_only_inputs_preserved=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
