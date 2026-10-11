# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Measure an ordered DMA completion token before extending late task handoff.

This private experiment reserves the last 64 bytes of L1/UB, outside the small
probe's allocations. It does not change the installed compiler or runtime.
The process-local wrapper hook is deliberately not an operator authoring API.
"""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import pypto.language as pl
import torch
from pypto.backend import pto_backend
from pypto.ir import TensorType
from pypto.ir._kernel_compile import kernel_abi_for_program
from pypto.ir.compile import _compile_impl
from pypto.pypto_core.passes import RuntimeKind
from pypto.runtime import RunConfig
from pypto.runtime._kernel_artifact import KernelArtifact


def program(cube):
    blocks = 24 if cube else 48
    tile_rows = 128 if cube else 32
    rows = blocks * tile_rows + 16

    @pl.jit
    def completion_probe(
        x: pl.Tensor[[rows, 256], pl.FP16],
        w: pl.Tensor[[128, 256], pl.FP16],
        out: pl.Out[pl.Tensor[[rows, 256], pl.FP32]],
        trace: pl.Out[pl.Tensor[[blocks + 1, 32], pl.INT64]],
        loops: pl.Scalar[pl.INT64],
    ):
        with pl.spmd(blocks, name_hint="completion_probe_task"):
            block = pl.tile.get_block_idx()
            pl.system.wait_pre_task_end()
            pl.tensor.write(trace, [block, 31], pl.cast(block, pl.INT64))
            if cube:
                a_l1 = pl.load(x, [block * tile_rows, 0], [tile_rows, 128], target_memory=pl.MemorySpace.Mat)
                b_l1 = pl.load(w, [0, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul(a, b)
                for _ in pl.range(loops):
                    acc = pl.matmul_acc(acc, a, b)
                pl.store(acc, [block * tile_rows, 0], out)
            else:
                value = pl.cast(pl.load(x, [block * tile_rows, 0], [tile_rows, 256]), pl.FP32)
                for _ in pl.range(loops):
                    value = pl.add(value, 1.0)
                pl.store(value, [block * tile_rows, 0], out)
            pl.system.set_next_task_start()
        return out

    return completion_probe


@contextmanager
def completion_adapter(mode, cube):
    original = pto_backend._generate_kernel_wrapper

    def wrap(func, *args, **kwargs):
        code = original(func, *args, **kwargs)
        assert func.name == "completion_probe_task", func.name
        tensor_params = [p for p in func.params if isinstance(p.type, TensorType)]
        trace_index = next(i for i, p in enumerate(tensor_params) if p.name_hint.startswith("trace"))
        out_index = next(i for i, p in enumerate(tensor_params) if p.name_hint.startswith("out"))
        tile_rows = 128 if cube else 32
        notification = ""
        if mode != "barrier":
            if cube:
                notification = """
    auto* scratch = reinterpret_cast<__cbuf__ uint16_t*>(524224);
    copy_gm_to_cbuf(scratch, reinterpret_cast<__gm__ uint16_t*>(row + 16), 0, 1, 2, 0, 0, (pad_t)0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_M, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_MTE3);
    copy_cbuf_to_gm(reinterpret_cast<__gm__ uint16_t*>(row + 8), scratch, 0, 1, 2, 0, 0);
"""
            else:
                notification = """
    auto* scratch = reinterpret_cast<__ubuf__ uint16_t*>(196544);
    copy_gm_to_ubuf(scratch, reinterpret_cast<__gm__ uint16_t*>(row + 16), 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_MTE3);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ uint16_t*>(row + 8), scratch, 0, 1, 2, 0, 0);
"""
        polling = ""
        if mode == "poll":
            polling = """
    // No AICPU change. Read only the existing task register, never SPR MMIO.
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    const uint32_t own_id = ctx->reg_task_id;
    for (;;) {
        uint32_t dispatched;
        asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(dispatched));
        // Values from platform_config.h, whose host-only includes do not
        // belong in this standalone generated probe.
        constexpr uint32_t idle_task = 0x7ffffffdu;
        constexpr uint32_t exit_signal = 0x7ffffff0u;
        const uint32_t next = dispatched;
        if (next != own_id && next != idle_task && next != exit_signal) {
            reason = 2;
            break;
        }
        dcci(row + 8, SINGLE_CACHE_LINE);
        if (*reinterpret_cast<volatile __gm__ uint64_t*>(row + 8) == 1) {
            reason = 1;
            break;
        }
        ++iterations;
        if (get_sys_cnt() - entered > 1000000) {
            reason = 3;
            break;
        }
    }
"""
        replacement = f"""
    const int32_t block = get_block_idx(args);
    auto* trace_tensor = reinterpret_cast<__gm__ Tensor*>(args[{trace_index}]);
    auto* row = reinterpret_cast<__gm__ uint64_t*>(trace_tensor->buffer.addr)
        + trace_tensor->start_offset + block * 32;
    const uint64_t entered = get_sys_cnt();
    uint64_t iterations = 0;
    uint64_t reason = 0;
    {notification}
    const uint64_t notified = get_sys_cnt();
    {polling}
    const uint64_t observed = get_sys_cnt();
    // Record one output word before the final full barrier. Host checks all
    // words afterwards; this word checks visibility at the notification.
    auto* output_tensor = reinterpret_cast<__gm__ Tensor*>(args[{out_index}]);
    auto* last = reinterpret_cast<__gm__ uint32_t*>(output_tensor->buffer.addr)
        + output_tensor->start_offset + (block + 1) * {tile_rows * 256} - 1;
    dcci(last, SINGLE_CACHE_LINE);
    const uint32_t snapshot = *reinterpret_cast<volatile __gm__ uint32_t*>(last);
    pipe_barrier(PIPE_ALL);
    const uint64_t drained = get_sys_cnt();
    row[0] = entered;
    row[1] = notified;
    row[2] = observed;
    row[3] = drained;
    row[4] = iterations;
    row[5] = reason;
    row[6] = snapshot;
    row[7] = get_coreid();
    dcci(row, SINGLE_CACHE_LINE, CACHELINE_OUT);
"""
        old = """    // The compiler transfers the ordinary tail PIPE_ALL to the executor.
    // No signal or mutable GM completion state is needed at this boundary.
    (void)args;
"""
        assert code.count(old) == 1, "local Set adapter changed"
        code = code.replace(old, replacement)
        # Explicit reset on every replay, before the small probe submits any
        # computation. Token and target occupy distinct 64-byte cache lines.
        trace_var = tensor_params[trace_index].name_hint
        reset = f"""    auto* __probe_row = reinterpret_cast<__gm__ uint64_t*>({trace_var})
        + get_block_idx(args) * 32;
    __probe_row[8] = 0;
    __probe_row[16] = 1;
    dcci(__probe_row + 8, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci(__probe_row + 16, SINGLE_CACHE_LINE, CACHELINE_OUT);
"""
        assert code.count("    // Forward to ptoas-generated function\n") == 1, "wrapper call changed"
        code = code.replace(
            "    // Forward to ptoas-generated function\n",
            reset + "    // Forward to ptoas-generated function\n",
        )
        return code

    pto_backend._generate_kernel_wrapper = wrap
    try:
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original


def artifact(cube, mode, directory, compile_only):
    blocks = 24 if cube else 48
    rows = blocks * (128 if cube else 32) + 16
    meta_args = (
        torch.empty((rows, 256), dtype=torch.float16, device="meta"),
        torch.empty((128, 256), dtype=torch.float16, device="meta"),
        torch.empty((rows, 256), dtype=torch.float32, device="meta"),
        torch.empty((blocks + 1, 32), dtype=torch.int64, device="meta"),
        1,
    )
    prog = program(cube).specialize(*meta_args)
    abi = kernel_abi_for_program(prog, platform="a2a3", runtime="tensormap_and_ringbuffer")
    if compile_only:
        opts = RunConfig(platform="a2a3").compile_kwargs()
        opts.update(output_dir=str(directory / "build"), runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER)
        with completion_adapter(mode, cube):
            result = _compile_impl(prog, _kernel_abi=abi, **opts)
    else:
        result = KernelArtifact(directory / "build", abi)
    result.load()
    return result


def main():  # noqa: PLR0912 -- small standalone experiment with explicit validation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int)
    options = parser.parse_args()
    modes = ("barrier", "notify", "poll")
    artifacts = {
        (cube, mode): artifact(
            cube, mode, options.compiled / f"{'aic' if cube else 'aiv'}_{mode}", options.compile
        )
        for cube in (False, True)
        for mode in modes
    }
    if options.compile:
        return
    assert os.environ.get("TASK_DEVICE") == str(options.device)
    import torch_npu  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    torch_npu.npu.set_device(options.device)
    init(device=options.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    runtime = get_process_kernel_state()
    registered = {key: runtime.ensure_callable(a, runtime.require_config()) for key, a in artifacts.items()}
    report = {"device": options.device, "scope": "single task, one block per core, no successor", "cases": []}
    options.output.mkdir(parents=True, exist_ok=True)
    try:
        for cube in (False, True):
            blocks = 24 if cube else 48
            tile_rows = 128 if cube else 32
            rows = blocks * tile_rows + 16
            cpu_x = (torch.arange(rows * 256) % 7 - 3).to(torch.float16).reshape(rows, 256)
            cpu_w = (torch.arange(128 * 256) % 5 - 2).to(torch.float16).reshape(128, 256)
            x, w = cpu_x.npu(options.device), cpu_w.npu(options.device)
            outputs = [torch.empty((rows, 256), device=x.device) for _ in modes]
            traces = [torch.full((blocks + 1, 32), -1, dtype=torch.int64, device=x.device) for _ in modes]
            for loops in (1, 64):
                graphs, correctness = [], []
                for mode, out, trace in zip(modes, outputs, traces):
                    registration = registered[cube, mode]
                    for changed in (0, 1):
                        x.copy_((cpu_x + changed).npu(options.device))
                        out.fill_(float("nan"))
                        trace.fill_(-1)
                        enqueue(registration, (x, w, out, trace, loops))
                        torch_npu.npu.synchronize()
                        actual_trace = trace.cpu().tolist()
                        # Retain raw evidence even when a following assertion fails.
                        (options.output / f"trace_{cube}_{mode}_{loops}_{changed}.json").write_text(
                            json.dumps(actual_trace) + "\n"
                        )
                        if cube:
                            expected = (cpu_x[: blocks * tile_rows, :128] + changed).float() @ cpu_w.float()
                            expected *= loops + 1
                        else:
                            expected = (cpu_x[: blocks * tile_rows] + changed).float() + loops
                        actual = out.cpu()
                        assert torch.equal(actual[: blocks * tile_rows], expected), (
                            cube,
                            mode,
                            loops,
                            "output",
                        )
                        assert torch.isnan(actual[blocks * tile_rows :]).all(), "output guard"
                        assert actual_trace[-1] == [-1] * 32, "trace guard"
                        for b, row in enumerate(actual_trace[:-1]):
                            assert 0 < row[0] <= row[1] <= row[2] <= row[3]
                            assert row[31] == b
                            if mode != "barrier":
                                assert row[8] == 1 and row[16] == 1, "DMA notification missing"
                            if mode == "poll":
                                assert row[5] == 1, ("unexpected poll exit", row)
                                last = (
                                    expected[(b + 1) * tile_rows - 1, -1].reshape(1).view(torch.int32).item()
                                )
                                assert row[6] == last & 0xFFFFFFFF, ("completion preceded output", b, row)
                        correctness.append(
                            {"mode": mode, "loops": loops, "changed_input": changed, "exact": True}
                        )
                    graph = torch_npu.npu.NPUGraph()
                    with torch_npu.npu.graph(graph):
                        enqueue(registration, (x, w, out, trace, loops))
                    graphs.append(graph)
                    for _ in range(5):
                        graph.replay()
                torch_npu.npu.synchronize()
                start, end = (torch_npu.npu.Event(enable_timing=True) for _ in range(2))
                samples = []
                for repeat in range(15):
                    for offset in range(3):
                        i = (repeat + offset) % 3
                        start.record()
                        graphs[i].replay()
                        end.record()
                        end.synchronize()
                        samples.append(
                            {"repeat": repeat, "mode": modes[i], "us": start.elapsed_time(end) * 1000}
                        )
                stats = {}
                for mode in modes:
                    values = [v["us"] for v in samples if v["mode"] == mode]
                    stats[mode] = {"min": min(values), "mean": mean(values), "max": max(values)}
                traces_final = {mode: trace.cpu().tolist() for mode, trace in zip(modes, traces)}
                case = {
                    "cube": cube,
                    "loops": loops,
                    "stats": stats,
                    "samples": samples,
                    "correctness": correctness,
                    "final_trace": traces_final,
                }
                report["cases"].append(case)
                (options.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"cube": cube, "loops": loops, "stats": stats}), flush=True)
                for graph in graphs:
                    graph.reset()
    finally:
        runtime.drain()
        runtime.close()


if __name__ == "__main__":
    main()
