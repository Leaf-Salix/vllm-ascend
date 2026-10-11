# SPDX-License-Identifier: Apache-2.0
"""Private diagnostic wrappers: no scheduler, task graph, or arithmetic edits.

The trace address is fixed only for the lifetime of this diagnostic process.
These artifacts must never be reused in another process or used for timing.
"""

from collections import Counter, defaultdict
from contextlib import contextmanager

MAX_RAW_CORES = 128
RECORDS_PER_CORE = 2048
WORDS_PER_RECORD = 16


def instrument_source(source, function_id, address, has_tail):
    helper = f"""
static __aicore__ __attribute__((always_inline)) inline
__gm__ uint64_t* csa_coverage_row(__gm__ int64_t* args) {{
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    uint64_t core = get_coreid();
    uint64_t sequence = ctx->reg_task_id;
    auto* arena = reinterpret_cast<__gm__ uint64_t*>({address}ULL);
    if (core >= {MAX_RAW_CORES}) return nullptr;
    if (sequence >= {RECORDS_PER_CORE}) {{
        auto* overflow = arena + core * {RECORDS_PER_CORE * WORDS_PER_RECORD};
        overflow[0] = core;
        overflow[1] = sequence;
        dcci(overflow, SINGLE_CACHE_LINE, CACHELINE_OUT);
        return nullptr;
    }}
    return arena + (core * {RECORDS_PER_CORE} + sequence) * {WORDS_PER_RECORD};
}}
"""
    adapter_marker = 'extern "C" {\n'
    assert adapter_marker in source
    source = source.replace(adapter_marker, helper + adapter_marker, 1)
    source = source.replace(
        "    int64_t ready = 1;\n",
        "    auto* coverage = csa_coverage_row(args);\n"
        "    if (coverage != nullptr) coverage[14] = get_sys_cnt();\n"
        "    int64_t ready = 1;\n",
        1,
    )
    after_wait = "    // The Scalar prefix must not change state inherited by the predecessor.\n"
    assert after_wait in source
    source = source.replace(
        after_wait,
        "    if (coverage != nullptr) {\n"
        "        coverage[15] = get_sys_cnt();\n"
        "        dcci(coverage + 8, SINGLE_CACHE_LINE, CACHELINE_OUT);\n"
        "    }\n" + after_wait,
        1,
    )
    boundary_decode = "    args = reinterpret_cast<__gm__ int64_t*>(__pypto_args_word & ~uint64_t{1});\n"
    assert source.count(boundary_decode) == 1
    setup = f"""
    auto* coverage_ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    auto* coverage_row = csa_coverage_row(args);
    if (coverage_row != nullptr) {{
        coverage_row[0] = get_coreid();
        coverage_row[1] = get_sys_cnt();
        coverage_row[8] = {function_id};
        coverage_row[9] = coverage_ctx->reg_task_id;
        coverage_row[10] = coverage_ctx->scalar_prefix;
        coverage_row[11] = coverage_ctx->predecessor_reg_task_id;
        coverage_row[12] = get_block_idx(args);
        coverage_row[13] = (coverage_ctx->wait_predecessor_fn != 0 ? 1ULL : 0ULL)
            | (coverage_ctx->predecessor_context != 0 ? 2ULL : 0ULL)
            | ({int(has_tail)}ULL << 2);
    }}
"""
    if has_tail:
        setup += """
    coverage_ctx->scalar_overlap_trace = reinterpret_cast<uint64_t>(coverage_row);
    if (coverage_row != nullptr) coverage_ctx->scalar_overlap_start = coverage_row[1];
"""
    source = source.replace(boundary_decode, boundary_decode + setup, 1)
    if not has_tail:
        # Synchronous members already drain their ordinary epilogue. Do not
        # dirty the runtime's overlap context for those two scalar-only entries.
        ending = """
    if (coverage_row != nullptr) {
        uint64_t finished = get_sys_cnt();
        coverage_row[2] = finished;
        coverage_row[3] = finished;
        coverage_row[4] = 0;
        coverage_row[5] = finished;
        coverage_row[6] = 0;
        coverage_row[7] = finished;
        dcci(coverage_row, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
"""
        assert source.endswith("}\n")
        source = source[:-2] + ending + "}\n"
    return source


@contextmanager
def instrument_wrappers(address, functions):
    from pypto.backend import pto_backend

    original = pto_backend._generate_kernel_wrapper

    def wrapper(func, *args, **kwargs):
        assert pto_backend._function_uses_ops(func, pto_backend._LOCAL_TASK_WAIT_OPS), func.name
        has_tail = pto_backend._function_uses_ops(func, pto_backend._LOCAL_TASK_SIGNAL_OPS)
        function_id = len(functions)
        source = original(func, *args, **kwargs)
        functions.append(
            {
                "id": function_id,
                "name": func.name,
                "has_tail": has_tail,
                "has_compiled_poll": "task_boundary_scratch_base" in dict(func.attrs),
                "blocking_tail_poll": f"// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 {func.name}\n" in source,
            }
        )
        return instrument_source(source, function_id, address, has_tail)

    pto_backend._generate_kernel_wrapper = wrapper
    try:
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original


def analyze_trace(tensor, functions):
    assert tensor.shape == (MAX_RAW_CORES, RECORDS_PER_CORE, WORDS_PER_RECORD)
    assert (tensor[:, 0, :] == -1).all(), "trace capacity exceeded or dispatch sequence zero"
    indices = (tensor[:, :, 8] >= 0).nonzero()
    records = [tensor[int(core), int(sequence)].tolist() for core, sequence in indices]
    names = {f["id"]: f["name"] for f in functions}
    blocking_poll = {f["id"]: f.get("blocking_tail_poll", False) for f in functions}
    prior = {(r[0], r[9]): r for r in records}
    total, by_function = Counter(), defaultdict(Counter)
    details, missed = [], []
    for row in records:
        core, begin, returned, finished, tail_flags, wait_begin, _, compute_end = row[:8]
        fn, sequence, flags, predecessor, block, mode, prefix_end, after_wait = row[8:]
        assert fn in names
        assert 0 < begin <= prefix_end <= after_wait <= returned <= finished, row
        assert returned <= wait_begin <= compute_end <= finished, row
        counts = by_function[names[fn]]
        counts["executions"] += 1
        counts["gated_dispatch"] += bool(flags & 4)
        ready_gate = bool(flags & 4) and not bool(mode & 1)
        counts["gated_ordinary_ready"] += ready_gate
        if ready_gate:
            # The executor's ordinary gate/EXIT checks precede ACK and entry.
            # This trace has no pre-ACK DMB timestamp, so it does not measure
            # when the gate opened. Local tails must still use their callback.
            assert not mode & 2, row
        counts["pending_capable"] += bool(flags & 1)
        counts["prefix_callback"] += bool(mode & 1)
        counts["local_handoff"] += bool(mode & 2)
        counts["pending_without_handoff"] += bool(flags & 1) and not bool(mode & 2)
        counts["idle_or_drained_prefix"] += bool(mode & 1) and not bool(mode & 2)
        notification = bool(tail_flags & 256)
        late_handoff = bool(tail_flags & 512)
        counts["notification_submitted"] += notification
        counts["handoff_after_callback"] += late_handoff
        counts["handoff_after_blocking_drain"] += late_handoff and blocking_poll[fn]
        counts["notification_without_handoff"] += notification and not late_handoff
        if notification:
            assert mode & 4 and tail_flags & 255 == 7, row
        if late_handoff:
            # Resource-exhausted functions can only hand off after their
            # callback's full drain. This is not evidence of compute overlap.
            assert notification or blocking_poll[fn], row
        if mode & 2:
            producer = prior.get((core, predecessor))
            assert producer is not None, ("missing predecessor", core, predecessor)
            assert producer[2] <= begin <= prefix_end <= producer[5] <= producer[3] <= after_wait
            by_function[names[producer[8]]]["outgoing_handoff"] += 1
            by_function[names[producer[8]]]["outgoing_without_callback"] += not bool(producer[4] & 512)
            counts["incoming_gated_handoff"] += bool(flags & 4)
            counts["incoming_ready_handoff"] += not bool(flags & 4)
            # The complete tail now includes notification DMA. A positive wait
            # cannot independently prove that the original compute overlapped.
            positive_tail = producer[7] - producer[5] > 5
            counts["handoff_with_tail_gt5ticks"] += positive_tail
            counts["received_handoff_after_callback"] += bool(producer[4] & 512)
            details.append(
                {
                    "core": core,
                    "sequence": sequence,
                    "function": names[fn],
                    "predecessor": names[producer[8]],
                    "block": block,
                    "prefix_ticks": prefix_end - begin,
                    "predecessor_tail_wait_ticks": producer[7] - producer[5],
                    "predecessor_submitted_notification": bool(producer[4] & 256),
                    "after_callback": bool(producer[4] & 512),
                }
            )
        elif flags & 1:
            producer = prior.get((core, predecessor))
            assert producer is not None, ("missing pending predecessor", core, predecessor)
            missed.append(
                {
                    "core": core,
                    "sequence": sequence,
                    "function": names[fn],
                    "predecessor": names[producer[8]],
                    "predecessor_has_tail": bool(producer[13] & 4),
                    "predecessor_submitted_notification": bool(producer[4] & 256),
                    "entry_after_predecessor_finish_ticks": begin - producer[3],
                }
            )
    for counts in by_function.values():
        total.update(counts)
    assert total["handoff_after_callback"] == total["received_handoff_after_callback"], total
    assert total["outgoing_handoff"] == total["local_handoff"], total
    return {
        "totals": dict(total),
        "by_function": {k: dict(v) for k, v in sorted(by_function.items())},
        "handoffs": details,
        "pending_without_handoff": missed,
        "records": records,
    }
