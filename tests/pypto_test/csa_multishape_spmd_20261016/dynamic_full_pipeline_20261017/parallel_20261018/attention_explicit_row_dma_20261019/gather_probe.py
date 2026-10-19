# SPDX-License-Identifier: Apache-2.0
"""复用候选inline helper验证行DMA、OOB置零与固定地址重放；不测性能。"""
import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from source_loader import load_package

probe_parser = argparse.ArgumentParser(add_help=False)
probe_parser.add_argument("output", type=Path)
probe_parser.add_argument("--spmd-source", type=Path, default=HERE / "source_candidate")
probe_args, _ = probe_parser.parse_known_args()
package = load_package(probe_args.spmd_source, "row_dma_probe")
stage = importlib.import_module(package.name + ".attention_stream_stages")
attention_gather_physical_rows = stage.attention_gather_physical_rows

import pypto.language as pl

TABLE_ROWS = stage.COMPRESSED_ROWS_DYN
PLAN_ROWS = stage.PLAN_CAP_DYN


@pl.jit.incore
def gather_core(
    table: pl.Tensor[[TABLE_ROWS, 512], pl.BF16],
    indices: pl.Tensor[[PLAN_ROWS, 512], pl.INT32],
    output: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
):
    first = pl.cast(0, pl.INDEX)
    gathered = attention_gather_physical_rows(table, indices, first, first)
    pl.store(gathered, [0, 0], output)
    return output


@pl.jit
def gather_entry(
    table: pl.Tensor[[TABLE_ROWS, 512], pl.BF16],
    indices: pl.Tensor[[PLAN_ROWS, 512], pl.INT32],
    output: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
):
    with pl.spmd(1):
        output = gather_core(table, indices, output)
    return output


def main():
    import torch
    import torch_npu
    import pypto.torch
    from pypto import passes
    from pypto.runtime import CompileOptions

    folder = probe_args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, device_execution=False, performance_measured=False,
                  full_csa_validated=False, helper_source=str(Path(stage.__file__).resolve()),
                  checks=[])
    try:
        did = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(did)
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device("npu", did)
        pypto.torch.init(device=did, platform="a2a3", runtime="tensormap_and_ringbuffer")
        config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        table_cpu = ((torch.arange(193).view(-1, 1) * 13 + torch.arange(512) * 3) % 127 - 63).to(torch.bfloat16)
        table = table_cpu.to(device)
        indices_cpu = torch.full((1, 512), -555, dtype=torch.int32)
        base = torch.arange(64, dtype=torch.int32) * 19 % 193
        indices_cpu[0, :64] = base
        mixed = base.flip(0).clone()
        mixed[::11] = -1
        mixed[3::17] = 193
        mixed[5::23] = 2147483647
        mixed[7::29] = -2147483648
        indices = indices_cpu.to(device)
        allocation = torch.full((66, 512), -777, dtype=torch.bfloat16, device=device)
        output = allocation[1:-1]
        pointers = [x.data_ptr() for x in (table, indices, output)]
        gather_entry(table, indices, output, config=config)
        torch.npu.synchronize()
        report["device_execution"] = True
        assert torch.equal(output.cpu(), table_cpu[base.long()])
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            gather_entry(table, indices, output, config=config)
        for label, values in (("valid", base), ("mixed", mixed),
                              ("all_invalid", torch.full_like(base, -1)), ("valid_again", base)):
            indices_cpu[0, :64] = values
            indices.copy_(indices_cpu.to(device))
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            valid = (values >= 0) & (values < 193)
            expected = torch.zeros((64, 512), dtype=torch.bfloat16)
            expected[valid] = table_cpu[values[valid].long()]
            actual = output.cpu()
            mismatches = int((actual != expected).sum())
            guards = bool((allocation.cpu()[[0, -1]] == -777).all())
            same_addresses = pointers == [x.data_ptr() for x in (table, indices, output)]
            inputs_unchanged = torch.equal(table.cpu(), table_cpu) and torch.equal(indices.cpu(), indices_cpu)
            report["checks"].append(dict(variant=label, mismatches=mismatches, guards=guards,
                                         same_addresses=same_addresses, inputs_unchanged=inputs_unchanged))
            assert mismatches == 0 and guards and same_addresses and inputs_unchanged
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
