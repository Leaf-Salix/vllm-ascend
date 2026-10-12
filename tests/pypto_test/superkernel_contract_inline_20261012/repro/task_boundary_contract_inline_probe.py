# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Inline cached overlap preparation without changing dispatch decisions."""

import task_boundary_contract_cache_probe as cached
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cached.changes(root)
    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp"
    old, new = result[path]
    anchor = "__attribute__((noinline)) void initialize_overlap_context("
    assert new.count(anchor) == 1
    result[path] = (
        old,
        new.replace(anchor, "__attribute__((always_inline)) inline void initialize_overlap_context("),
    )
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
