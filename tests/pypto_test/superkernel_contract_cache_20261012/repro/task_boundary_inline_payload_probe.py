# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Inline dispatch payload preparation without changing dispatch decisions."""

import task_boundary_cold_init_probe as cold
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cold.changes(root)
    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp"
    old, new = result[path]
    anchor = "bool SchedulerContext::build_payload("
    assert new.count(anchor) == 1
    new = new.replace(anchor, "__attribute__((always_inline, used)) inline bool SchedulerContext::build_payload(")
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
