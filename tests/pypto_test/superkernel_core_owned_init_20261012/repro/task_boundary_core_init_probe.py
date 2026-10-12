# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Move mutable overlap initialization off the serial dispatch preparer."""

import task_boundary_contract_cache_probe as cached
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cached.changes(root)
    runtime = root / "src/a2a3/runtime/tensormap_and_ringbuffer"
    path = runtime / "common/intrinsic.h"
    old = path.read_text()
    anchor = "    uint64_t overlap_cache_padding[2];"
    assert old.count(anchor) == 1
    new = old.replace(anchor, """    // Written before publication by AICPU; AICore never modifies this line.
    uint64_t submitted_overlap;
    uint64_t submitted_predecessor;""")
    anchor = "// Independent bits: entry ABI never grants early execution."
    assert new.count(anchor) == 1
    new = new.replace(anchor, """// Consume one published generation before a wrapper or handoff reads it.
__aicore__ __attribute__((always_inline)) inline void initialize_submitted_overlap(
    __gm__ GlobalContext *ctx, uint32_t reg_task_id
) {
    const uint64_t state = ctx->submitted_overlap;
    const uint64_t previous = ctx->submitted_predecessor;
    ctx->async_tail_kind = static_cast<uint32_t>(state);
    ctx->scalar_prefix = static_cast<uint32_t>(state >> 32);
    ctx->late_handoff_scratch = static_cast<uint32_t>(previous);
    ctx->predecessor_reg_task_id = static_cast<uint32_t>(previous >> 32);
    ctx->reg_task_id = reg_task_id;
    ctx->scalar_overlap_trace = 0;
    ctx->wait_predecessor_fn = 0;
    ctx->predecessor_context = 0;
}

""" + anchor)
    result[path] = (old, new)

    path = runtime / "runtime/scheduler/scheduler_dispatch.cpp"
    old, new = result[path]
    start = new.index("    overlap.async_tail_kind = contract.tail_kind;")
    end = new.index("    overlap.scalar_prefix = prefix;", start) + len("    overlap.scalar_prefix = prefix;")
    new = new[:start] + """    const uint32_t scratch = core_exec_state.dispatch_id_reused ? contract.scratch :
        contract.scratch & ~COMPILED_POLL_ID_REUSED;
    overlap.submitted_overlap = (uint64_t{prefix} << 32) | contract.tail_kind;
    overlap.submitted_predecessor = (uint64_t{predecessor_id} << 32) | scratch;
    (void)reg_task_id;""" + new[end:]
    result[path] = (old, new)

    path = runtime / "aicore/aicore_executor.cpp"
    old, new = result[path]
    anchor = """    } else if (trace_addr != 0 ||
               (kind != AsyncTailKind::RuntimeBarrierAll && kind != AsyncTailKind::RuntimeCompiledPollClean &&
                kind != AsyncTailKind::RuntimeDirectPollClean)) {"""
    assert new.count(anchor) == 1
    new = new.replace(anchor, """    } else {
        // Pickup initialized this line on AICore; retire it before FIN permits
        // slot reuse or a following invocation's cold initialization.""")
    anchor = """    if ((next->global_context.scalar_prefix & SCALAR_PREFIX_CAN_HANDOFF) == 0 ||"""
    assert new.count(anchor) == 1
    new = new.replace(anchor, """    initialize_submitted_overlap(&next->global_context, next_id);
""" + anchor)
    anchor = """            if (task_id != prepared_prefix_task_id) dcci(exec_payload, ENTIRE_DATA_CACHE);
            prepared_prefix_task_id = AICPU_IDLE_TASK_ID;
            prepared_predecessor = nullptr;
            const bool ordinary =
                exec_payload->local_context.async_ctx.local_predecessor_pending == DISPATCH_ORDINARY_ENTRY;"""
    assert new.count(anchor) == 1
    new = new.replace(anchor, """            const bool prepared_handoff = task_id == prepared_prefix_task_id;
            if (!prepared_handoff) dcci(exec_payload, ENTIRE_DATA_CACHE);
            prepared_prefix_task_id = AICPU_IDLE_TASK_ID;
            prepared_predecessor = nullptr;
            const bool ordinary =
                exec_payload->local_context.async_ctx.local_predecessor_pending == DISPATCH_ORDINARY_ENTRY;
            if (!ordinary && !prepared_handoff) {
                initialize_submitted_overlap(&exec_payload->global_context, task_id);
            }""")
    start = new.index("            } else if (scalar_prefix_handoff) {")
    end = new.index("\n\n            // Keep start_time", start)
    new = new[:start] + """            } else if (!ordinary) {
                flush_overlap_context(&exec_payload->global_context);
            }""" + new[end:]
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
