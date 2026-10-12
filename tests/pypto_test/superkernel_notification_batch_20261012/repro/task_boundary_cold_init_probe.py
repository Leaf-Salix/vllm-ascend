# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Keep proven ordinary dispatches on the original compact scheduler path."""

import subprocess

import task_boundary_hot_path_probe as hot
import task_boundary_normal_state_probe as normal


def changes(root):
    result = hot.changes(root)
    base = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler"
    del result[base / "scheduler_types.h"]
    path = base / "scheduler_dispatch.cpp"
    old = path.read_text()
    original = subprocess.check_output(
        ["git", "show", f"ef6812891575eaf2a15051fc9abc659f5415c2e2:{path.relative_to(root)}"],
        cwd=root, text=True,
    )
    start = old.index("    auto &overlap = payload.global_context;")
    end = old.index("\n    if (to_pending) {", start)
    helper = """
// Proven synchronous entries never consume this state. Every other entry
// initializes it before publication, including after an ordinary slot reuse.
__attribute__((noinline)) void initialize_overlap_context(
    DispatchPayload &payload, ChipTaskSlotState &slot_state, CoreExecState &core_exec_state,
    SubtaskSlot subslot, uint32_t reg_task_id, bool to_pending, bool force_gate
) {
""" + old[start:end] + "}\n"
    anchor = "\n}\n\n// The early-dispatch core bitmask"
    assert original.count(anchor) == 1
    new = original.replace(anchor, helper + anchor)
    anchor = "    if (reg_task_id >= AICORE_EXIT_SIGNAL) {\n"
    assert new.count(anchor) == 1
    new = new.replace(anchor, anchor + "        core_exec_state.dispatch_id_reused = true;\n")
    anchor = "\n    if (to_pending) {"
    assert new.count(anchor) == 1
    new = new.replace(anchor, """
    const uint8_t member = 1u << static_cast<uint32_t>(subslot);
    const auto &source_contract = *slot_state.payload;
    const bool ordinary =
        (source_contract.scalar_prefix_tagged_args & (member << 3)) != 0 &&
        ((source_contract.scalar_prefix | source_contract.runtime_barrier_tail |
          source_contract.runtime_compiled_poll) & member) == 0;
    payload.local_context.async_ctx.local_predecessor_pending = ordinary ? DISPATCH_ORDINARY_ENTRY : 0u;
    if (!ordinary) {
        initialize_overlap_context(payload, slot_state, core_exec_state, subslot, reg_task_id,
                                   to_pending, force_gate);
    }
""" + anchor)
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
