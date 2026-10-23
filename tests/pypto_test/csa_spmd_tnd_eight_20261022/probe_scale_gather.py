# SPDX-License-Identifier: Apache-2.0
"""Device-only diagnostic of the exact scale helper; no CSA performance claim."""

import argparse
import json
import os
import traceback
from pathlib import Path

import pypto.language as pl
from scale_gather_source import gather_scale_pages

CACHE_PAGES = 41
PAGE_BYTES = 4160
BYTE_ROWS = CACHE_PAGES * PAGE_BYTES // 64


@pl.jit.incore
def scale_probe_kernel(
    cache: pl.Tensor[[BYTE_ROWS, 64], pl.INT8],
    table: pl.Tensor[[3, 65], pl.INT32],
    control: pl.Tensor[[3], pl.INT32],
    output: pl.InOut[pl.Tensor[[1, 1024], pl.FP16]],
):
    if pl.tile.get_subblock_idx() == 0:
        batch = pl.cast(pl.read(control, [0]), pl.INDEX)
        begin = pl.cast(pl.read(control, [1]), pl.INDEX)
        length = pl.cast(pl.read(control, [2]), pl.INDEX)
        scales = gather_scale_pages(cache, table, batch, begin, length, CACHE_PAGES, PAGE_BYTES, 1024)
        output = pl.store(scales, [0, 0], output)
    return output


@pl.jit
def scale_probe(
    cache: pl.Tensor[[BYTE_ROWS, 64], pl.INT8],
    table: pl.Tensor[[3, 65], pl.INT32],
    control: pl.Tensor[[3], pl.INT32],
    output: pl.InOut[pl.Tensor[[1, 1024], pl.FP16]],
):
    with pl.spmd(1):
        output = scale_probe_kernel(cache, table, control, output)
    return output


def main():
    import numpy as np
    import pypto.torch
    import torch
    from pypto import passes
    from pypto.runtime import CompileOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, performance_eligible=False, device=args.device, checks=[])
    try:
        options = CompileOptions(
            platform="a2a3", output_dir=str(args.output / "kernel"), memory_planner=passes.MemoryPlanner.PYPTO
        )
        if args.compile_only:
            arguments = dict(
                cache=torch.empty((BYTE_ROWS, 64), dtype=torch.int8, device="meta"),
                table=torch.empty((3, 65), dtype=torch.int32, device="meta"),
                control=torch.empty(3, dtype=torch.int32, device="meta"),
                output=torch.empty((1, 1024), dtype=torch.float16, device="meta"),
                config=options,
            )
            artifact = scale_probe._resolve_kernel_artifact((), arguments)
            report.update(passed=True, device_execution=False, artifact=str(artifact.output_dir))
            return
        import torch_npu  # noqa: F401 - register NPU only for this queued diagnostic

        assert args.device == int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(args.device)
        pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        rng = np.random.default_rng(1024)
        raw = rng.integers(0, 256, size=(CACHE_PAGES, PAGE_BYTES), dtype=np.uint8)
        table = np.stack(
            [np.arange(65) % CACHE_PAGES, np.arange(64, -1, -1) % CACHE_PAGES, np.arange(65) % CACHE_PAGES]
        ).astype(np.int32)
        table[2, ::3] = -1
        table[2, 1::7] = np.iinfo(np.int32).max
        table[2, 2::11] = CACHE_PAGES
        cache_npu = torch.from_numpy(raw.reshape(BYTE_ROWS, 64).view(np.int8)).to(f"npu:{args.device}")
        table_npu = torch.from_numpy(table).to(f"npu:{args.device}")
        output = torch.empty((1, 1024), dtype=torch.float16, device=f"npu:{args.device}")
        control = torch.empty(3, dtype=torch.int32, device=f"npu:{args.device}")
        for batch, begin, length in [(0, 0, 1024), (0, 1024, 1025), (2, 0, 1024), (1, 2048, 2080), (2, -32, 1024)]:
            report["active_case"] = [batch, begin, length]
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            control.copy_(torch.tensor([batch, begin, length], dtype=torch.int32))
            output.view(torch.uint8).fill_(37)
            scale_probe(cache_npu, table_npu, control, output, config=options)
            torch.npu.synchronize()
            expected = np.zeros(2048, dtype=np.uint8)
            for lane in range(32):
                logical = min(begin // 32 + lane, max((length - 1) // 32, 0))
                if 0 <= logical < table.shape[1]:
                    physical = int(table[batch, logical])
                    if 0 <= physical < CACHE_PAGES:
                        expected[lane * 64 : (lane + 1) * 64] = raw[physical, 4096:4160]
            actual = output.cpu().view(torch.uint8).numpy().reshape(-1)
            mismatch = np.flatnonzero(actual != expected)
            record = dict(control=[batch, begin, length], passed=not len(mismatch), mismatch_bytes=len(mismatch))
            if len(mismatch):
                artifact = args.output / f"case_{len(report['checks'])}_bytes.npz"
                np.savez(artifact, actual=actual, expected=expected)
                record.update(artifact=str(artifact), first_mismatch_positions=mismatch[:32].tolist())
            report["checks"].append(record)
        report["passed"] = all(row["passed"] for row in report["checks"])
        assert report["passed"], "Device helper byte mismatch; all per-case outputs retained"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
