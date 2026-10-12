# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Retire core-owned control cache lines once at the end of the dispatch loop."""

import task_boundary_core_init_probe as core_init
import task_boundary_normal_state_probe as normal


def changes(root):
    result = core_init.changes(root)
    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    old, new = result[path]
    anchor = """    if (!runtime_tail) {
        ctx->async_tail_kind = 0;
        flush_overlap_context(ctx);
    } else {
        // Pickup initialized this line on AICore; retire it before FIN permits
        // slot reuse or a following invocation's cold initialization.
        flush_overlap_context(ctx);
    }"""
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """    if (!runtime_tail) ctx->async_tail_kind = 0;
    // Only this physical core writes the control line. AICPU publishes the
    // next generation in the separate immutable descriptor line. Diagnostic
    // records still become visible at the original per-task boundary.
    if (trace_addr != 0) flush_overlap_context(ctx);""",
    )
    anchor = """    // The control line must be clean before FIN permits AICPU buffer reuse.
    flush_overlap_context(ctx);"""
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """    // AICPU never overwrites this core-owned line during dispatch-slot reuse.
    if (ctx->scalar_overlap_trace != 0) flush_overlap_context(ctx);""",
    )
    anchor = """            } else if (!ordinary) {
                flush_overlap_context(&exec_payload->global_context);
            }"""
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """            } else if (!ordinary && exec_payload->global_context.scalar_overlap_trace != 0) {
                flush_overlap_context(&exec_payload->global_context);
            }""",
    )
    anchor = "    bool exiting = false;"
    assert new.count(anchor) == 1
    new = new.replace(anchor, anchor + "\n    bool overlap_context_used = false;")
    anchor = """            if (!ordinary && !prepared_handoff) {
                initialize_submitted_overlap(&exec_payload->global_context, task_id);
            }"""
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """            if (!ordinary) {
                overlap_context_used = true;
                if (!prepared_handoff) initialize_submitted_overlap(&exec_payload->global_context, task_id);
            }""",
    )
    # Both EXIT paths must publish the control lines before AICPU can retire
    # their backing allocation. This includes EXIT observed inside the gate.
    anchor = "write_reg(RegId::COND, AICORE_EXITED_VALUE);"
    assert new.count(anchor) == 2
    new = new.replace(
        anchor,
        """if (overlap_context_used) {
                flush_overlap_context(&payload[0].global_context);
                flush_overlap_context(&payload[1].global_context);
            }
            write_reg(RegId::COND, AICORE_EXITED_VALUE);""",
    )
    result[path] = old, new
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
