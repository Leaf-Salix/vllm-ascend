# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Keep the original entry when a gated task's complete input gate is already open."""

import task_boundary_contract_cache_probe as cached
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cached.changes(root)
    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    old, new = result[path]
    start = new.index("                if (!ordinary && !scalar_prefix_handoff")
    loop = "                while (!scalar_prefix_handoff) {"
    stop = new.index(loop, start) + len(loop)
    anchor = new[start:stop]
    assert new.count(anchor) == 1
    new = new.replace(
        anchor,
        """                bool input_gate_ready = false;
                if (!ordinary && !scalar_prefix_handoff && !chip_swimlane_enabled &&
                    !pmu_enabled && !dump_args_enabled &&
                    (exec_payload->global_context.scalar_prefix & SCALAR_PREFIX_CAN_EARLY_START) != 0) {
                    // The local tail was already drained. A matching full-fanin
                    // doorbell needs no delayed Wait or mutable callback state.
                    // Preserve the ordinary gate's EXIT check, without polling
                    // the same matching doorbell again in the loop below.
                    input_gate_ready = read_dmb_high32() == task_id;
                    if (input_gate_ready) {
                        exiting = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) ==
                                  AICORE_EXIT_SIGNAL;
                    } else {
                        exec_payload->global_context.wait_predecessor_fn =
                            reinterpret_cast<uint64_t>(&wait_scalar_prefix_predecessor);
                        if ((exec_payload->global_context.scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) == 0) {
                            exec_payload->local_context.async_ctx.local_predecessor_pending = 1;
                        }
                        scalar_prefix_handoff = true;
                    }
                }
                while (!scalar_prefix_handoff && !input_gate_ready) {""",
    )
    result[path] = old, new
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
