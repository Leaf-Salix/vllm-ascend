# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Diagnostic only: quantify late-notification cost in the fixed task chain.

This variant retains immediately available handoffs but misses successors that
arrive later. It must not be retained as an equivalent implementation.
"""

import task_boundary_contract_cache_probe as cached
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cached.changes(root)
    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    old, new = result[path]
    start = new.index("                    if (next_id == AICPU_IDLE_TASK_ID &&")
    end = new.index("                    if (next_id != AICPU_IDLE_TASK_ID)", start)
    removed = new[start:end]
    assert "poll_late_handoff(payload" in removed
    assert "poll_late_handoff_direct(payload" in removed
    new = new[:start] + new[end:]
    new = new.replace(
        "__attribute__((noinline)) static uint32_t poll_late_handoff",
        "__attribute__((noinline, unused)) static uint32_t poll_late_handoff",
    )
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
