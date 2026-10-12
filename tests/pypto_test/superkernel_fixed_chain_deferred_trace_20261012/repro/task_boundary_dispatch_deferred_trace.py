# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Keep diagnostic cache-clean barriers out of the dispatch loop."""

import task_boundary_dispatch_trace as trace

original_changes = trace.changes


def changes(root):
    result = original_changes(root)
    base = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler"
    path = base / "scheduler_context.h"
    old, new = result[path]
    anchor = "        wmb();\n        const uint64_t publish_after = get_sys_cnt_aicpu();"
    assert new.count(anchor) == 1
    new = new.replace(anchor, "        const uint64_t publish_after = get_sys_cnt_aicpu();")
    anchor = "            cache_flush_range(h.boundary_trace, 64);\n"
    assert new.count(anchor) == 1
    new = new.replace(anchor, "")
    anchor = "    CoreTracker core_trackers_[MAX_AICPU_THREADS];"
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        anchor
        + """
    struct alignas(64) BoundaryTraceRoot {
        uint64_t *address{nullptr};
    };
    BoundaryTraceRoot boundary_trace_roots_[MAX_AICPU_THREADS];
""",
    )
    result[path] = old, new
    path = base / "scheduler_dispatch.cpp"
    old, new = result[path]
    anchor = "            trace[0] = prepare_begin;"
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """            if (boundary_trace_roots_[thread_idx].address == nullptr) {
                boundary_trace_roots_[thread_idx].address = reinterpret_cast<uint64_t *>(tensor.buffer.addr);
            }
"""
        + anchor,
    )
    anchor = "int32_t SchedulerContext::resolve_and_dispatch(Runtime *runtime, int32_t thread_idx) {"
    assert new.count(anchor) == 1
    new = new.replace(anchor, anchor + "\n    boundary_trace_roots_[thread_idx].address = nullptr;")
    anchor = "    // Drain any entries left in the deferred-release batch."
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """    // Diagnostic only: all fixed-chain dispatches are over. Each thread
    // cleans its own cached stores; no per-publish barrier perturbs the chain.
    if (boundary_trace_roots_[thread_idx].address != nullptr) {
        cache_flush_range(boundary_trace_roots_[thread_idx].address, 1632 * 8 * sizeof(uint64_t));
        boundary_trace_roots_[thread_idx].address = nullptr;
    }

"""
        + anchor,
    )
    result[path] = old, new
    return result


if __name__ == "__main__":
    trace.changes = changes
    trace.main()
