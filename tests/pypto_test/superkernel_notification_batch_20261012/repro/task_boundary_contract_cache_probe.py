# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Derive immutable member contracts once at task submission, before publication."""

import task_boundary_compact_init_probe as compact
import task_boundary_normal_state_probe as normal


def changes(root):
    result = compact.changes(root)
    base = root / 'src/a2a3/runtime/tensormap_and_ringbuffer/runtime'
    path = base / 'runtime_types.h'
    old = path.read_text()
    anchor = '    alignas(64) DispatchPredicate predicate;'
    assert old.count(anchor) == 1
    fields = '''
    struct OverlapContract {
        uint32_t tail_kind{0};
        uint32_t prefix{0};
        uint32_t scratch{0};
    };
    // Submit-owned immutable descriptors fit in the predicate line padding.
    // Bit 16 identifies a proven synchronous entry; bit 17 advertises a Wait.
    OverlapContract overlap_contracts[SUBTASK_SLOT_COUNT]{};

    void prepare_overlap_contracts() {
        for (int index = 0; index < SUBTASK_SLOT_COUNT; ++index) {
            const uint8_t member = 1u << index;
            const bool tail = (runtime_barrier_tail & member) != 0;
            const bool poll = (runtime_compiled_poll & member) != 0;
            const uint32_t scratch = (runtime_direct_poll_scratch >> (index == 0 ? 0 : 16)) & 0xffffu;
            const bool direct = tail && poll && scratch != 0;
            auto &contract = overlap_contracts[index];
            contract.tail_kind = static_cast<uint32_t>(!tail ? AsyncTailKind::None :
                direct ? AsyncTailKind::RuntimeDirectPollClean :
                poll ? AsyncTailKind::RuntimeCompiledPollClean : AsyncTailKind::RuntimeBarrierAll);
            contract.scratch = direct ? (scratch - 1u) * 64u : poll ? COMPILED_POLL_ID_REUSED : 0u;
            uint32_t prefix = (scalar_prefix_tagged_args & member) != 0 ? SCALAR_PREFIX_TAGGED_ARGS : 0u;
            if (prefix && (scalar_prefix_tagged_args & SCALAR_PREFIX_SOURCE_ARGS_CAPABILITY)) {
                prefix |= SCALAR_PREFIX_SOURCE_ARGS;
            }
            if (scalar_prefix & member) prefix |= 1u << 17;
            if ((scalar_prefix_tagged_args & (member << 3)) != 0 &&
                ((scalar_prefix | runtime_barrier_tail | runtime_compiled_poll) & member) == 0) {
                prefix |= 1u << 16;
            }
            contract.prefix = prefix;
        }
    }
'''
    new = old.replace(anchor, anchor + fields)
    result[path] = (old, new)
    path = base / 'orchestrator.cpp'
    old = path.read_text()
    anchor = '    out->payload->runtime_direct_poll_scratch = args.runtime_direct_poll_scratch();'
    assert old.count(anchor) == 1
    result[path] = (old, old.replace(anchor, anchor+'\n    out->payload->prepare_overlap_contracts();'))

    path = base / 'scheduler/scheduler_dispatch.cpp'
    old, new = result[path]
    start = new.index('    const uint8_t member = 1u << static_cast<uint32_t>(subslot);')
    end = new.index('    uint32_t predecessor_id = 0;', start)
    new = new[:start] + '''    const auto &contract = source.overlap_contracts[static_cast<uint32_t>(subslot)];
    uint32_t prefix = contract.prefix & 0xffffu;
''' + new[end:]
    new = new.replace('    if ((source.scalar_prefix & member) != 0) {',
                      '    if ((contract.prefix & (1u << 17)) != 0) {',1)
    start = new.index('    overlap.async_tail_kind = static_cast<uint32_t>')
    end = new.index('    overlap.scalar_overlap_trace = 0;', start)
    new = new[:start]+'''    overlap.async_tail_kind = contract.tail_kind;
    overlap.late_handoff_scratch = core_exec_state.dispatch_id_reused ? contract.scratch :
        contract.scratch & ~COMPILED_POLL_ID_REUSED;
'''+new[end:]
    start = new.index('    const uint8_t member = 1u << static_cast<uint32_t>(subslot);', new.index('SchedulerContext::PublishHandle'))
    end = new.index('    payload.local_context.async_ctx.local_predecessor_pending', start)
    new = new[:start]+'''    const auto &contract = slot_state.payload->overlap_contracts[static_cast<uint32_t>(subslot)];
    const bool ordinary = (contract.prefix & (1u << 16)) != 0;
'''+new[end:]
    result[path] = (old, new)
    return result


if __name__ == '__main__':
    normal.changes = changes
    normal.main()
