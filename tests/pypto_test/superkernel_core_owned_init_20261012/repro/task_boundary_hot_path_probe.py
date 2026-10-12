# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Isolated experiment: skip overlap metadata for proven ordinary entries."""

import task_boundary_normal_state_probe as normal


_normal_changes = normal.changes


def changes(root):
    result = _normal_changes(root)
    runtime = root / "src/a2a3/runtime/tensormap_and_ringbuffer"
    path = runtime / "runtime/dispatch_payload.h"
    old = path.read_text()
    anchor = "struct alignas(64) DispatchPayload {"
    assert old.count(anchor) == 1
    result[path] = (old, old.replace(anchor, """// A compiler-proven synchronous entry does not consume overlap metadata.
// This value occupies the existing hot local-context flag only on such entries.
inline constexpr uint32_t DISPATCH_ORDINARY_ENTRY = 2u;

""" + anchor))

    path = runtime / "runtime/scheduler/scheduler_dispatch.cpp"
    old, new = result[path]
    anchor = "    const uint8_t buffer_bit = 1u << buf_idx;"
    assert new.count(anchor) == 1
    new = new.replace(anchor, """    payload.local_context.async_ctx.local_predecessor_pending = ordinary ? DISPATCH_ORDINARY_ENTRY : 0u;
""" + anchor)
    result[path] = (old, new)

    path = runtime / "aicore/aicore_executor.cpp"
    old = path.read_text()
    replacements = {
        "    dcci(next, ENTIRE_DATA_CACHE);\n": """    dcci(next, ENTIRE_DATA_CACHE);
    if (next->local_context.async_ctx.local_predecessor_pending == DISPATCH_ORDINARY_ENTRY)
        return AICPU_IDLE_TASK_ID;
""",
        "    __gm__ DispatchPayload *payload, __gm__ int64_t *args\n":
            "    __gm__ DispatchPayload *payload, __gm__ int64_t *args, bool ordinary\n",
        "    if (payload->global_context.scalar_overlap_trace != 0) {":
            "    if (!ordinary && payload->global_context.scalar_overlap_trace != 0) {",
        "            bool scalar_prefix_handoff = exec_payload->global_context.predecessor_context != 0;": """            const bool ordinary =
                exec_payload->local_context.async_ctx.local_predecessor_pending == DISPATCH_ORDINARY_ENTRY;
            bool scalar_prefix_handoff = !ordinary && exec_payload->global_context.predecessor_context != 0;""",
        "                source_args = !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&":
            "                source_args = !ordinary && !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&",
        "                if (!scalar_prefix_handoff && !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&":
            "                if (!ordinary && !scalar_prefix_handoff && !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&",
        "            execute_task(exec_payload, call_args);":
            "            execute_task(exec_payload, call_args, ordinary);",
        "            const auto tail_kind = static_cast<AsyncTailKind>(exec_payload->global_context.async_tail_kind);": """            const auto tail_kind = ordinary ? AsyncTailKind::None :
                static_cast<AsyncTailKind>(exec_payload->global_context.async_tail_kind);""",
    }
    new = old
    for anchor, replacement in replacements.items():
        assert new.count(anchor) == 1, anchor
        new = new.replace(anchor, replacement)
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
