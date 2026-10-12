# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Verify enabled scheduler timing still records a complete fixed 5 us chain."""

import argparse
import json
import os
import sys
from pathlib import Path

from simpler_setup.runtime_builder import RuntimeBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = Path(os.environ["SCALAR_OVERLAP_ROOT"])
    sys.path.insert(0, str(private / "pypto/tests/st/runtime/kernel"))
    import task_boundary_acceptance as acceptance  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import begin_dfx, end_dfx, init  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    RuntimeBuilder._CACHE_DIR = args.root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = args.root / "runtime/lib"
    device = int(os.environ["TASK_DEVICE"])
    torch.set_num_threads(2)
    torch_npu.npu.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    init(
        device=device,
        runtime="tensormap_and_ringbuffer",
        ring_task_window=128,
        enable_chip_swimlane=4,
        output_dir=output,
    )
    runtime = get_process_kernel_state()
    acceptance.SCALAR_STEPS = 256
    try:
        kernel = runtime.ensure_callable(
            acceptance.artifact(
                private / "artifacts/boundary_compact_sdk_20261012/compiled", "plain"
            ),
            runtime.require_config(),
        )
        data = torch.empty((acceptance.ROWS, acceptance.COLS), device=f"npu:{device}")
        trace = torch.full((817, 8), -1, dtype=torch.int64, device=data.device)
        source = (torch.arange(48 * 32 * 512, dtype=torch.float32) % 31).reshape(
            -1, 512
        )
        data.fill_(float("nan"))
        data[:1536].copy_(source)
        begin_dfx()
        enqueue(kernel, (data, trace, 17, 5, 256, 7))
        end_dfx()
        acceptance.verify(data, source, 5, 17)
        assert torch.equal(trace.cpu(), torch.full((817, 8), -1, dtype=torch.int64))
        recorded = json.loads((output / "chip_swimlane_records.json").read_text())
        assert recorded["chip_swimlane_level"] == 4
        assert len(recorded["aicore_tasks"]) == 17 * 48
        records = [
            r
            for stream in recorded["scheduler_records"]["streams"]
            for r in stream["records"]
        ]
        assert records
        assert all(0 < r["start_cycles"] <= r["end_cycles"] for r in records)
        nonempty = [r for r in records if r["end_cycles"] > r["start_cycles"]]
        assert nonempty
        result = {
            "status": "PASS",
            "device": device,
            "aicore_tasks": 17 * 48,
            "scheduler_records": len(records),
            "nonempty_intervals": len(nonempty),
        }
        (output / "check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        runtime.drain()
        runtime.close()


if __name__ == "__main__":
    main()
