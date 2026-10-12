# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Bounded TMR acceptance: unmarked overhead and calibrated 5/50 us tasks."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean, median

import torch
from pypto.ir._kernel_compile import kernel_abi_for_program
from pypto.ir.compile import _compile_impl
from pypto.pypto_core.passes import RuntimeKind
from pypto.runtime import RunConfig
from pypto.runtime._kernel_artifact import KernelArtifact

BLOCKS = 48
TILE_ROWS = 32
COLS = 512
STAGES = 17
ROWS = (STAGES + 1) * BLOCKS * TILE_ROWS + 16
SCALAR_STEPS = 16
TICKS_PER_US = 50


_PROGRAM = """import pypto.language as pl

@pl.jit
def acceptance_chain(
    data: pl.InOut[pl.Tensor[[27664, 512], pl.FP32]],
    trace: pl.Out[pl.Tensor[[817, 8], pl.INT64]],
    length: pl.Scalar[pl.INT64],
    work: pl.Scalar[pl.INT64],
    scalar_steps: pl.Scalar[pl.INT64],
    seed: pl.Scalar[pl.INT64],
):
    for stage in pl.range(length):
        with pl.spmd(48, allow_early_resolve=True, name_hint="acceptance_task"):
            block = pl.tile.get_block_idx()
            # DIAG_BEGIN
            state = seed + stage + block
            for _ in pl.range(scalar_steps):
                state = (state * 17 + 3) % 65521
            # WAIT
            value = pl.load(data, [(stage * 48 + block) * 32, 0], [32, 512])
            value = pl.add(value, pl.cast(state % 17, pl.FP32))
            for _ in pl.range(work):
                value = pl.add(value, 1.0)
            pl.store(value, [((stage + 1) * 48 + block) * 32, 0], data)
            # SET
    return data, trace
"""


def make_program(root, mode):
    source = _PROGRAM
    if mode == "early":
        source = source.replace("# WAIT", "pl.system.wait_pre_task_end()")
        source = source.replace("# SET", "pl.system.set_next_task_start()")
    if mode == "calibrate":
        source = source.replace(
            "# DIAG_BEGIN", "pl.system.bar_all()\n            started = pl.system.read_clock()"
        )
        source = source.replace("# WAIT", "scalar_end = pl.system.read_clock(state)")
        source = source.replace(
            "# SET",
            """pl.system.bar_all()
            finished = pl.system.read_clock()
            pl.tensor.write(trace, [stage * 48 + block, 0], started)
            pl.tensor.write(trace, [stage * 48 + block, 1], scalar_end)
            pl.tensor.write(trace, [stage * 48 + block, 2], finished)""",
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "program.py"
    if path.exists():
        assert path.read_text() == source, f"frozen source changed: {path}"
    else:
        path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"acceptance_{mode}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.acceptance_chain.specialize()


def artifact(root, mode, compile_now=False):
    program = make_program(root / mode, mode)
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="tensormap_and_ringbuffer")
    output = root / mode
    if compile_now:
        options = RunConfig(platform="a2a3").compile_kwargs()
        options.update(output_dir=str(output), runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER)
        compiled = _compile_impl(program, _kernel_abi=abi, **options)
    else:
        compiled = KernelArtifact(output, abi)
    compiled.load()
    return compiled


def stats(values):
    return {"min_us": min(values), "mean_us": mean(values), "max_us": max(values)}


def verify(data, source, work, length, seed=7):
    actual = data.cpu()
    expected = source.clone()
    assert torch.equal(actual[: BLOCKS * TILE_ROWS], source)
    for stage in range(length):
        for block in range(BLOCKS):
            state = seed + stage + block
            for _ in range(SCALAR_STEPS):
                state = (state * 17 + 3) % 65521
            region = slice(block * TILE_ROWS, (block + 1) * TILE_ROWS)
            expected[region].add_(work + state % 17)
        start = (stage + 1) * BLOCKS * TILE_ROWS
        assert torch.equal(actual[start : start + BLOCKS * TILE_ROWS], expected), f"stage {stage}"
    assert torch.isnan(actual[(length + 1) * BLOCKS * TILE_ROWS :]).all(), "guard modified"


def run(args):  # noqa: PLR0912 -- one bounded calibration and timing entry
    assert os.environ.get("TASK_DEVICE") == str(args.device), "use queue-assigned card"
    import torch_npu  # noqa: PLC0415 -- only initialize framework runtime on device
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    torch.set_num_threads(2)
    torch_npu.npu.set_device(args.device)
    init(device=args.device, runtime="tensormap_and_ringbuffer", ring_task_window=128)
    runtime = get_process_kernel_state()
    modes = ["calibrate"] if args.action == "calibrate" else args.modes
    graphs = {}
    report = {
        "device": args.device,
        "task_blocks": BLOCKS,
        "stages": STAGES,
        "scalar_steps": SCALAR_STEPS,
        "replays_per_graph": args.replays_per_graph,
        "ticks_per_us": TICKS_PER_US,
        "samples": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    data = torch.empty((ROWS, COLS), dtype=torch.float32, device=f"npu:{args.device}")
    trace = torch.empty((STAGES * BLOCKS + 1, 8), dtype=torch.int64, device=data.device)
    source = (torch.arange(BLOCKS * TILE_ROWS * COLS, dtype=torch.float32) % 31).reshape(-1, COLS)

    def reset(shift=0):
        data.fill_(float("nan"))
        data[: BLOCKS * TILE_ROWS].copy_(source + shift)
        trace.fill_(-1)

    try:
        regs = {
            mode: runtime.ensure_callable(artifact(args.compiled, mode), runtime.require_config())
            for mode in modes
        }
        if args.action == "calibrate":
            records = []

            def measure(work):
                reset()
                enqueue(regs["calibrate"], (data, trace, 1, work, SCALAR_STEPS, 7))
                torch_npu.npu.synchronize()
                rows = trace[:BLOCKS, :3].cpu().tolist()
                task = [(v[2] - v[0]) / TICKS_PER_US for v in rows]
                scalar = [(v[1] - v[0]) / TICKS_PER_US for v in rows]
                compute = [(v[2] - v[1]) / TICKS_PER_US for v in rows]
                assert all(t > 0 for t in task)
                records.append(
                    {
                        "work": work,
                        "task": stats(task),
                        "scalar": stats(scalar),
                        "compute": stats(compute),
                        "ticks": rows,
                    }
                )
                return median(task)

            measure(1)  # Warm caches before calibration.
            low = measure(1)
            high = measure(64)
            slope = (high - low) / 63
            assert slope > 0, "work count did not increase duration"
            selected = []
            for target in (5.0, 50.0):
                work = max(1, round(1 + (target - low) / slope))
                duration = measure(work)
                if abs(duration - target) > target * 0.15:
                    work = max(1, round(work + (target - duration) / slope))
                    duration = measure(work)
                assert abs(duration - target) <= target * 0.25, (target, duration)
                verify(data, source, work, 1)
                selected.append({"target_us": target, "work": work, "measured": records[-1]})
            report.update(selected=selected, calibration=records, status="PASS")
        else:
            calibration = json.loads(args.calibration.read_text())
            workloads = [(row["target_us"], row["work"]) for row in calibration["selected"]]
            for target, work in workloads:
                for length in (1, STAGES):
                    for mode in modes:
                        print(
                            json.dumps(
                                {"phase": "initial_enqueue", "work": work, "length": length, "mode": mode}
                            ),
                            flush=True,
                        )
                        reset()
                        call_args = (data, trace, length, work, SCALAR_STEPS, 7)
                        enqueue(regs[mode], call_args)
                        torch_npu.npu.synchronize()
                        verify(data, source, work, length)
                        graph = torch_npu.npu.NPUGraph()
                        with torch_npu.npu.graph(graph):
                            for _ in range(args.replays_per_graph):
                                enqueue(regs[mode], call_args)
                        graphs[target, length, mode] = graph
                        for _ in range(5):
                            graph.replay()
                        torch_npu.npu.synchronize()
                    for repeat in range(30):
                        for offset in range(len(modes)):
                            mode = modes[(repeat + offset) % len(modes)]
                            start = torch_npu.npu.Event(enable_timing=True)
                            end = torch_npu.npu.Event(enable_timing=True)
                            start.record()
                            graphs[target, length, mode].replay()
                            end.record()
                            end.synchronize()
                            report["samples"].append(
                                {
                                    "target_us": target,
                                    "work": work,
                                    "length": length,
                                    "mode": mode,
                                    "repeat": repeat,
                                    "us": start.elapsed_time(end) * 1000 / args.replays_per_graph,
                                }
                            )
                    for mode in modes:
                        reset(11)
                        graphs[target, length, mode].replay()
                        torch_npu.npu.synchronize()
                        verify(data, source + 11, work, length)
            report["summary"] = []
            for target, work in workloads:
                for length in (1, STAGES):
                    for mode in modes:
                        samples = [
                            s["us"]
                            for s in report["samples"]
                            if (s["target_us"], s["length"], s["mode"]) == (target, length, mode)
                        ]
                        report["summary"].append(
                            {
                                "target_us": target,
                                "work": work,
                                "length": length,
                                "mode": mode,
                                **stats(samples),
                            }
                        )
            report.update(status="PASS", output_and_guards_exact=True)
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Device teardown can fail too; preserve the original error and every
        # completed timing sample before entering framework cleanup.
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            for graph in reversed(list(graphs.values())):
                graph.reset()
            runtime.drain()
            runtime.close()
        except BaseException as error:
            report.update(status="FAIL", teardown_error=f"{type(error).__name__}: {error}")
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            raise
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "calibration")}, indent=2))


def main():
    global SCALAR_STEPS  # noqa: PLW0603 -- CLI workload shared by launch and exact golden
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compile", "calibrate", "run"))
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--modes", nargs="+", default=["plain", "early", "calibrate"])
    parser.add_argument("--scalar-steps", type=int, default=16)
    parser.add_argument("--replays-per-graph", type=int, default=1)
    args = parser.parse_args()
    SCALAR_STEPS = args.scalar_steps
    if args.action == "compile":
        for mode in args.modes:
            artifact(args.compiled, mode, compile_now=True)
            print(f"COMPILED {mode}", flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
