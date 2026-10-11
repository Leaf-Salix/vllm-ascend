# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check late same-core handoff without modifying AICPU dispatch or ACK flow."""

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
    tile_rows = 128 if cube else 64
    rows = blocks * tile_rows + 16

    @pl.jit
    def late_arrival_probe(  # noqa: PLR0913 -- independent data and diagnostic buffers
        x: pl.Tensor[[rows, 256], pl.FP16],
        w: pl.Tensor[[128, 256], pl.FP16],
        a_out: pl.Out[pl.Tensor[[rows, 256], pl.FP32]],
        b_out: pl.Out[pl.Tensor[[rows, 256], pl.FP32]],
        a_trace: pl.Out[pl.Tensor[[blocks + 1, 32], pl.INT64]],
        b_trace: pl.Out[pl.Tensor[[blocks + 1, 16], pl.INT64]],
        loops: pl.Scalar[pl.INT64],
        steps: pl.Scalar[pl.INT64],
        seed: pl.Scalar[pl.INT64],
        clock_modulus: pl.Scalar[pl.INT64],
    ):
        with pl.spmd(blocks, name_hint="late_producer", allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            pl.system.wait_pre_task_end()
            pl.tensor.write(a_trace, [block, 23], pl.cast(block, pl.INT64))
            if cube:
                a_l1 = pl.load(x, [block * tile_rows, 0], [tile_rows, 128], target_memory=pl.MemorySpace.Mat)
                b_l1 = pl.load(w, [0, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul(a, b)
                for _ in pl.range(loops):
                    acc = pl.matmul_acc(acc, a, b)
                pl.store(acc, [block * tile_rows, 0], a_out)
            else:
                value = pl.cast(pl.load(x, [block * tile_rows, 0], [tile_rows, 256]), pl.FP32)
                for _ in pl.range(loops):
                    value = pl.add(value, 1.0)
                pl.store(value, [block * tile_rows, 0], a_out)
            pl.system.set_next_task_start()
        with pl.spmd(blocks, name_hint="late_consumer", allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            core = pl.system.physical_core_id()
            begin = pl.system.read_clock()
            state = seed + pl.cast(block, pl.INT64) + begin % clock_modulus
            for _ in pl.range(steps):
                state = (state * 17 + 3) % 65521
            end = pl.system.read_clock(state)
            pl.system.wait_pre_task_end()
            after_wait = pl.system.read_clock()
            # Keep a real fan-in edge. Observing this value is forbidden before
            # Wait even when the physical predecessor is another block.
            observed = pl.tensor.read(a_out, [block * tile_rows, 0])
            if cube:
                a2_l1 = pl.load(x, [block * tile_rows, 0], [tile_rows, 128], target_memory=pl.MemorySpace.Mat)
                b2_l1 = pl.load(w, [0, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                a2 = pl.move(a2_l1, target_memory=pl.MemorySpace.Left)
                b2 = pl.move(b2_l1, target_memory=pl.MemorySpace.Right)
                acc2 = pl.matmul(a2, b2)
                pl.store(acc2, [block * tile_rows, 0], b_out)
            else:
                value2 = pl.cast(pl.load(x, [block * tile_rows, 0], [tile_rows, 256]), pl.FP32)
                pl.store(pl.add(value2, pl.cast(state, pl.FP32)), [block * tile_rows, 0], b_out)
            pl.tensor.write(b_trace, [block, 0], core)
            pl.tensor.write(b_trace, [block, 1], begin)
            pl.tensor.write(b_trace, [block, 2], end)
            pl.tensor.write(b_trace, [block, 3], after_wait)
            pl.tensor.write(b_trace, [block, 4], state)
            pl.tensor.write(b_trace, [block, 5], pl.cast(observed, pl.INT64))
        return a_out, b_out

    return late_arrival_probe


_CLOCK = r"""
static __aicore__ __attribute__((always_inline)) inline uint64_t probe_clock() {
    asm volatile("" : : : "memory");
    uint64_t tick = get_sys_cnt();
    asm volatile("" : "+l"(tick) : : "memory");
    return tick;
}
static __aicore__ __attribute__((always_inline)) inline uint32_t probe_next() {
    uint32_t task;
    asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(task) : : "memory");
    return task;
}
static __aicore__ __attribute__((always_inline)) inline bool probe_pending(uint32_t id, uint32_t own) {
    return id != own && id != 0x7ffffffdu && id != 0x7ffffff0u;
}
"""


def notification(cube):
    if cube:
        return """
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
    return """
    auto* scratch = reinterpret_cast<__ubuf__ uint16_t*>(196544);
    copy_gm_to_ubuf(scratch, reinterpret_cast<__gm__ uint16_t*>(row + 16), 0, 1, 2, 0, 0);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_MTE3);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ uint16_t*>(row + 8), scratch, 0, 1, 2, 0, 0);
"""


@contextmanager
def instrument(mode, cube):
    original = pto_backend._generate_kernel_wrapper

    def wrap(func, *args, **kwargs):
        code = original(func, *args, **kwargs)
        tensors = [p for p in func.params if isinstance(p.type, TensorType)]
        trace_name = "a_trace" if func.name == "late_producer" else "b_trace"
        trace_index = next(i for i, p in enumerate(tensors) if p.name_hint.startswith(trace_name))
        trace_var = tensors[trace_index].name_hint
        call_marker = "    // Forward to ptoas-generated function\n"
        assert code.count(call_marker) == 1
        code = code.replace('extern "C" {', _CLOCK + '\nextern "C" {', 1)
        if func.name == "late_producer":
            # Diagnostic-only compute marker: distinguish original M/V work
            # from the notification's added MTE3 tail. Existing executor kinds
            # record compute completion separately and consume every event.
            compute_marker = f"set_flag({'PIPE_M' if cube else 'PIPE_V'}, PIPE_S, EVENT_ID7);"
            if cube:
                completion_markers = """
    set_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    ctx->async_tail_kind = static_cast<uint32_t>(AsyncTailKind::AicAllPipes);
"""
            else:
                completion_markers = """
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    ctx->async_tail_kind = static_cast<uint32_t>(AsyncTailKind::AivAllPipes);
"""
            polling = ""
            if mode == "late":
                helper = f"""
// A cold noinline helper keeps DMA setup and polling live ranges out of
// the original computation and the already-present successor fast path.
static __aicore__ __attribute__((noinline)) uint64_t probe_wait_notification(
    __gm__ uint64_t* row, __gm__ uint64_t* negative, uint32_t own) {{
    const uint64_t start = probe_clock();
    dcci(row + 8, SINGLE_CACHE_LINE);
    const uint64_t previous = *reinterpret_cast<volatile __gm__ uint64_t*>(row + 8);
    const uint64_t expected = previous == 1 ? ~uint64_t{{0}} : 1;
    auto* source = expected == 1 ? row + 16 : negative;
    {notification(cube).replace("row + 16", "source")}
    uint32_t iterations = 0;
    uint32_t last = own;
    uint32_t reason = 0;
    for (;;) {{
        last = probe_next();
        if (probe_pending(last, own)) {{ reason = 2; break; }}
        dcci(row + 8, SINGLE_CACHE_LINE);
        if (*reinterpret_cast<volatile __gm__ uint64_t*>(row + 8) == expected) {{ reason = 3; break; }}
        ++iterations;
        if (last == 0x7ffffff0u) {{ reason = 5; break; }}
        if (probe_clock() - start > 1000000) {{ reason = 4; break; }}
    }}
    return (uint64_t{{last}} << 32) | ((uint64_t{{iterations}} & 0xffffffu) << 8) | reason;
}}
"""
                code = code.replace('extern "C" {', helper + '\nextern "C" {', 1)
                polling = f"""
    if (reason == 0) {{
        auto* negative = reinterpret_cast<__gm__ uint64_t*>(tensor->buffer.addr)
            + tensor->start_offset + {(24 if cube else 48) * 32};
        const uint64_t result = probe_wait_notification(row, negative, own);
        reason = result & 0xffu;
        iterations = (result >> 8) & 0xffffffu;
        last = result >> 32;
    }}
"""
            replacement = f"""
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[{trace_index}]);
    auto* row = reinterpret_cast<__gm__ uint64_t*>(tensor->buffer.addr)
        + tensor->start_offset + get_block_idx(args) * 32;
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    const uint32_t own = ctx->reg_task_id;
    const uint64_t start = probe_clock();
    const uint32_t initial = probe_next();
    uint32_t last = initial;
    uint64_t reason = probe_pending(initial, own) ? 1 : 0;
    uint64_t iterations = 0;
    {compute_marker}
    {polling}
    {completion_markers}
    dcci(reinterpret_cast<__gm__ char*>(ctx) + offsetof(GlobalContext, async_tail_kind),
         SINGLE_CACHE_LINE, CACHELINE_OUT);
    row[0] = start;
    row[1] = probe_clock();
    row[2] = initial;
    row[3] = own;
    row[4] = probe_pending(initial, own);
    row[5] = reason;
    row[6] = iterations;
    row[7] = last;
    dcci(row, SINGLE_CACHE_LINE, CACHELINE_OUT);
    // No FIN or ACK here. The existing executor validates the successor and
    // the original Wait callback drains PIPE_ALL before any resource reuse.
"""
            old = """    // The compiler transfers the ordinary tail PIPE_ALL to the executor.
    // No signal or mutable GM completion state is needed at this boundary.
    (void)args;
"""
            assert code.count(old) == 1
            code = code.replace(old, replacement)
            setup = f"""    auto* row = reinterpret_cast<__gm__ uint64_t*>({trace_var})
        + get_block_idx(args) * 32;
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    ctx->scalar_overlap_trace = reinterpret_cast<uint64_t>(row + 24);
    ctx->scalar_overlap_start = probe_clock();
"""
            return code.replace(call_marker, setup + call_marker)
        assert func.name == "late_consumer", func.name
        setup = """    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    const uint64_t own = ctx->reg_task_id;
    const uint64_t prior = ctx->predecessor_reg_task_id;
    const uint64_t flags = ctx->scalar_prefix;
    const uint64_t local_handoff = ctx->predecessor_context != 0;
"""
        code = code.replace(call_marker, setup + call_marker)
        insertion = f"""
    auto* row = reinterpret_cast<__gm__ uint64_t*>({trace_var}) + get_block_idx(args) * 16;
    row[8] = own;
    row[9] = prior;
    row[10] = flags;
    row[11] = local_handoff;
    dcci(row + 8, SINGLE_CACHE_LINE, CACHELINE_OUT);
"""
        end = code.rfind("}\n")
        return code[:end] + insertion + code[end:]

    pto_backend._generate_kernel_wrapper = wrap
    try:
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original


def artifact(cube, mode, directory, compile_only):
    blocks = 24 if cube else 48
    rows = blocks * (128 if cube else 64) + 16
    meta = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    prog = program(cube).specialize(
        meta((rows, 256), torch.float16),
        meta((128, 256), torch.float16),
        meta((rows, 256), torch.float32),
        meta((rows, 256), torch.float32),
        meta((blocks + 1, 32), torch.int64),
        meta((blocks + 1, 16), torch.int64),
        1,
        64,
        7,
        1,
    )
    abi = kernel_abi_for_program(prog, platform="a2a3", runtime="tensormap_and_ringbuffer")
    if compile_only:
        opts = RunConfig(platform="a2a3").compile_kwargs()
        opts.update(output_dir=str(directory / "build"), runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER)
        with instrument(mode, cube):
            result = _compile_impl(prog, _kernel_abi=abi, **opts)
    else:
        result = KernelArtifact(directory / "build", abi)
    result.load()
    return result


def validate(a, b, expected_a, expected_b, a_trace, b_trace, tile_rows, seed, steps):
    blocks = len(a_trace) - 1
    assert torch.equal(a[: blocks * tile_rows], expected_a), "producer output"
    assert torch.equal(b[: blocks * tile_rows], expected_b), "consumer output"
    assert torch.isnan(a[blocks * tile_rows :]).all() and torch.isnan(b[blocks * tile_rows :]).all()
    assert a_trace[-1] == [-1] * 32 and b_trace[-1] == [-1] * 16
    producer = {(v[24], v[3]): (i, v) for i, v in enumerate(a_trace[:-1])}
    assert len(producer) == blocks
    handoffs, late, overlaps = [], [], []
    for i, v in enumerate(a_trace[:-1]):
        assert 0 < v[25] <= v[0] <= v[1] <= v[26] <= v[29] <= v[31] <= v[27]
        assert v[28] == 7 and v[5] not in (4, 5), ("timeout/exit", i, v)
    for i, v in enumerate(b_trace[:-1]):
        state = seed + i
        for _ in range(steps):
            state = (state * 17 + 3) % 65521
        assert v[4] == state and v[5] == int(expected_a[i * tile_rows, 0])
        assert 0 < v[1] <= v[2] <= v[3]
        if not v[11]:
            continue
        ai, prev = producer[v[0], v[9]]
        assert v[1] >= prev[26] and v[3] >= prev[27], "incorrect local boundary order"
        entry = {
            "producer": ai,
            "consumer": i,
            "reason": prev[5],
            "remaining_ticks": prev[30],
            "remaining_compute_ticks": prev[31] - prev[29],
        }
        handoffs.append(entry)
        if prev[5] == 2:
            assert prev[4] == 0 and prev[2] == prev[3], "successor was already present"
            late.append(entry)
            if v[2] <= prev[29] and prev[31] - prev[29] > 5:
                overlaps.append(entry)
    return {"handoffs": handoffs, "late_handoffs": late, "late_with_remaining_compute": overlaps}


def main():  # noqa: PLR0912 -- standalone bounded experiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--work", type=int, nargs="+", default=[1, 64])
    args = parser.parse_args()
    modes = ("once", "late")
    artifacts = {
        (cube, mode): artifact(cube, mode, args.compiled / f"{'aic' if cube else 'aiv'}_{mode}", args.compile)
        for cube in (False, True)
        for mode in modes
    }
    if args.compile:
        return
    assert os.environ.get("TASK_DEVICE") == str(args.device)
    import torch_npu  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415

    torch_npu.npu.set_device(args.device)
    init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
    runtime = get_process_kernel_state()
    registrations = {
        key: runtime.ensure_callable(a, runtime.require_config()) for key, a in artifacts.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"device": args.device, "cases": []}
    try:
        for cube in (False, True):
            blocks, tile_rows = (24, 128) if cube else (48, 64)
            rows = blocks * tile_rows + 16
            cpu_x = (torch.arange(rows * 256) % 7 - 3).to(torch.float16).reshape(rows, 256)
            cpu_w = (torch.arange(128 * 256) % 5 - 2).to(torch.float16).reshape(128, 256)
            x, w = cpu_x.npu(args.device), cpu_w.npu(args.device)
            buffers = [
                (
                    torch.empty((rows, 256), device=x.device),
                    torch.empty((rows, 256), device=x.device),
                    torch.empty((blocks + 1, 32), dtype=torch.int64, device=x.device),
                    torch.empty((blocks + 1, 16), dtype=torch.int64, device=x.device),
                )
                for _ in modes
            ]
            for loops in args.work:
                case = {"cube": cube, "loops": loops, "steps": args.steps, "rounds": [], "samples": []}
                graphs = []
                for mode, (a, b, at, bt) in zip(modes, buffers):
                    registration = registrations[cube, mode]
                    for changed in (0, 1):
                        seed, steps = 7 + changed, args.steps
                        x.copy_((cpu_x + changed).npu(args.device))
                        a.fill_(float("nan"))
                        b.fill_(float("nan"))
                        at.fill_(-1)
                        # Immutable +1 / -1 source lines for the alternating
                        # notification. The final guard row supplies -1.
                        at[:blocks, 16].fill_(1)
                        bt.fill_(-1)
                        enqueue(registration, (x, w, a, b, at, bt, loops, steps, seed, 1))
                        torch_npu.npu.synchronize()
                        atr, btr = at.cpu().tolist(), bt.cpu().tolist()
                        raw = {"mode": mode, "changed": changed, "a": atr, "b": btr}
                        (args.output / f"trace_{cube}_{loops}_{mode}_{changed}.json").write_text(
                            json.dumps(raw) + "\n"
                        )
                        if cube:
                            expected_b = (cpu_x[: blocks * tile_rows, :128] + changed).float() @ cpu_w.float()
                            expected_a = expected_b * (loops + 1)
                        else:
                            expected_a = (cpu_x[: blocks * tile_rows] + changed).float() + loops
                            expected_b = (cpu_x[: blocks * tile_rows] + changed).float()
                            for block in range(blocks):
                                state = seed + block
                                for _ in range(steps):
                                    state = (state * 17 + 3) % 65521
                                expected_b[block * tile_rows : (block + 1) * tile_rows] += state
                        result = validate(
                            a.cpu(), b.cpu(), expected_a, expected_b, atr, btr, tile_rows, seed, steps
                        )
                        case["rounds"].append({"mode": mode, "changed": changed, "exact": True, **result})
                        print(
                            json.dumps(
                                {
                                    "cube": cube,
                                    "loops": loops,
                                    "mode": mode,
                                    "changed": changed,
                                    **{k: len(v) for k, v in result.items()},
                                }
                            ),
                            flush=True,
                        )
                    graph = torch_npu.npu.NPUGraph()
                    with torch_npu.npu.graph(graph):
                        enqueue(registration, (x, w, a, b, at, bt, loops, steps, seed, 1))
                    graphs.append(graph)
                    for _ in range(5 if args.repeats else 1):
                        graph.replay()
                torch_npu.npu.synchronize()
                start, end = (torch_npu.npu.Event(enable_timing=True) for _ in range(2))
                for repeat in range(args.repeats):
                    for offset in range(2):
                        i = (repeat + offset) % 2
                        start.record()
                        graphs[i].replay()
                        end.record()
                        end.synchronize()
                        case["samples"].append(
                            {"repeat": repeat, "mode": modes[i], "us": start.elapsed_time(end) * 1000}
                        )
                case["stats"] = {}
                for mode in modes:
                    values = [s["us"] for s in case["samples"] if s["mode"] == mode]
                    if values:
                        case["stats"][mode] = {"min": min(values), "mean": mean(values), "max": max(values)}
                case["replay_validation"] = []
                for mode, (a, b, at, bt) in zip(modes, buffers):
                    atr, btr = at.cpu().tolist(), bt.cpu().tolist()
                    (args.output / f"replay_{cube}_{loops}_{mode}.json").write_text(
                        json.dumps({"a": atr, "b": btr}) + "\n"
                    )
                    result = validate(
                        a.cpu(), b.cpu(), expected_a, expected_b, atr, btr, tile_rows, seed, steps
                    )
                    assert all(row[16] == 1 and row[8] in (-1, 1) for row in atr[:-1])
                    case["replay_validation"].append({"mode": mode, "exact": True, **result})
                report["cases"].append(case)
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                for graph in graphs:
                    graph.reset()
    finally:
        runtime.drain()
        runtime.close()


if __name__ == "__main__":
    main()
