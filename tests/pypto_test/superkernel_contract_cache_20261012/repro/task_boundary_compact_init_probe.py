# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Initialize overlap metadata once per field, preserving publication rules."""

import task_boundary_inline_payload_probe as inline
import task_boundary_normal_state_probe as normal


def changes(root):
    result = inline.changes(root)
    path = root / 'src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp'
    old, new = result[path]
    start = new.index('    auto &overlap = payload.global_context;')
    end = new.index('\n}\n\n}', start)
    replacement = '''    auto &overlap = payload.global_context;
    const auto &source = *slot_state.payload;
    const uint8_t member = 1u << static_cast<uint32_t>(subslot);
    const uint8_t tagged = source.scalar_prefix_tagged_args;
    const bool runtime_tail = (source.runtime_barrier_tail & member) != 0;
    const bool compiled_poll = (source.runtime_compiled_poll & member) != 0;
    const uint32_t scratch_contract = (source.runtime_direct_poll_scratch >>
        (subslot == SubtaskSlot::AIC ? 0 : 16)) & 0xffffu;
    const bool direct_poll = runtime_tail && compiled_poll && scratch_contract != 0;
    uint32_t prefix = (tagged & member) != 0 ? SCALAR_PREFIX_TAGGED_ARGS : 0u;
    if (prefix != 0 && (tagged & SCALAR_PREFIX_SOURCE_ARGS_CAPABILITY) != 0) {
        prefix |= SCALAR_PREFIX_SOURCE_ARGS;
    }
    uint32_t predecessor_id = 0;
    if ((source.scalar_prefix & member) != 0) {
        const bool gated_prefix = force_gate && source.fanin_wait_count > 0;
        const bool local_predecessor = to_pending && core_exec_state.running_slot_state != nullptr;
        if (gated_prefix) prefix |= SCALAR_PREFIX_CAN_EARLY_START;
        if (local_predecessor && (gated_prefix || payload.src_payload == 0)) {
            prefix |= SCALAR_PREFIX_CAN_HANDOFF;
            predecessor_id = static_cast<uint32_t>(core_exec_state.running_reg_task_id);
        }
    }
    overlap.async_tail_kind = static_cast<uint32_t>(!runtime_tail ? AsyncTailKind::None :
        direct_poll ? AsyncTailKind::RuntimeDirectPollClean :
        compiled_poll ? AsyncTailKind::RuntimeCompiledPollClean : AsyncTailKind::RuntimeBarrierAll);
    overlap.late_handoff_scratch = direct_poll ? (scratch_contract - 1u) * 64u :
        compiled_poll && core_exec_state.dispatch_id_reused ? COMPILED_POLL_ID_REUSED : 0u;
    overlap.scalar_overlap_trace = 0;
    overlap.wait_predecessor_fn = 0;
    overlap.predecessor_context = 0;
    overlap.predecessor_reg_task_id = predecessor_id;
    overlap.reg_task_id = reg_task_id;
    overlap.scalar_prefix = prefix;'''
    result[path] = (old, new[:start] + replacement + new[end:])
    return result


if __name__ == '__main__':
    normal.changes = changes
    normal.main()
