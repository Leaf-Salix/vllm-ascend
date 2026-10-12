# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Combine core-owned control with the already-ready input-gate entry."""

import task_boundary_core_owned_flush_probe as owned
import task_boundary_normal_state_probe as normal
import task_boundary_ready_gate_probe as ready


def changes(root):
    result = owned.changes(root)
    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    old, new = result[path]
    gate = ready.changes(root)[path][1]
    begin = gate.index("                bool input_gate_ready = false;")
    final = "                while (!scalar_prefix_handoff && !input_gate_ready) {"
    end = gate.index(final, begin) + len(final)
    begin_old = new.index("                if (!ordinary && !scalar_prefix_handoff")
    final_old = "                while (!scalar_prefix_handoff) {"
    end_old = new.index(final_old, begin_old) + len(final_old)
    new = new[:begin_old] + gate[begin:end] + new[end_old:]
    result[path] = old, new
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
