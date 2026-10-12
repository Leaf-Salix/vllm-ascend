// CPU-only tests for the experimental per-member ordinary-entry proof.
TEST_F(TmrSchedulerExecutionInputsTest, OrdinarySkipsColdStateAndEveryMarkedOrUnknownEntryReinitializes) {
    for (int core_offset = 0; core_offset < 3; ++core_offset) {
        for (bool marked : {false, true}) {
            initialize({table_a.data(), table_a.size()});
            const auto subslot = static_cast<SubtaskSlot>(core_offset);
            // Visit both buffers as ordinary, potentially dirty, then ordinary.
            for (int phase = 0; phase < 5; ++phase) {
                for (int buffer = 0; buffer < 2; ++buffer) {
                    init_slot();
                    slot.active_mask = ActiveMask(SUBTASK_MASK_ALL);
                    const bool ordinary = phase % 2 == 0;
                    payload.scalar_prefix_tagged_args = ordinary ? 56 : 0;
                    if (!ordinary && marked) {
                        payload.runtime_barrier_tail = SUBTASK_MASK_ALL;
                        payload.scalar_prefix = SUBTASK_MASK_ALL;
                    }
                    const auto handle = context->prepare_subtask_to_core(
                        0, core_offset, slot, subslot, false, 0, false);
                    ASSERT_TRUE(handle.valid);
                    auto &ctx = context->payload_per_core_[core_offset][handle.reg_task_id & 1u].global_context;
                    EXPECT_EQ(context->payload_per_core_[core_offset][handle.reg_task_id & 1u].local_context.async_ctx.local_predecessor_pending, ordinary ? DISPATCH_ORDINARY_ENTRY : 0u);
                    if (!ordinary) {
                        EXPECT_EQ(ctx.predecessor_context, 0u);
                        EXPECT_EQ(ctx.predecessor_reg_task_id, 0u);
                        EXPECT_EQ(ctx.wait_predecessor_fn, 0u);
                        EXPECT_EQ(ctx.scalar_overlap_trace, 0u);
                        // A legacy wrapper may write the same fields without
                        // declaring a modern local Wait/Set contract.
                        ctx.async_tail_kind = static_cast<uint32_t>(AsyncTailKind::RuntimeBarrierAll);
                        ctx.scalar_prefix = SCALAR_PREFIX_CAN_HANDOFF;
                        ctx.predecessor_context = 64;
                        ctx.predecessor_reg_task_id = 27;
                        ctx.wait_predecessor_fn = 128;
                        ctx.scalar_overlap_trace = 192;
                        ctx.late_handoff_scratch = ~uint64_t{0};
                    }
                }
            }
        }
    }
}

TEST_F(TmrSchedulerExecutionInputsTest, OrdinaryProofIsPerMemberAndCannotOverrideActiveContract) {
    for (int core_offset = 0; core_offset < 3; ++core_offset) {
        const auto subslot = static_cast<SubtaskSlot>(core_offset);
        const uint8_t member = 1u << core_offset;
        for (int contract = 0; contract < 4; ++contract) {
            initialize({table_a.data(), table_a.size()});
            // First populate the cache for both ordinary buffers, then offer
            // either another member's proof or a contradictory real contract.
            for (int phase = 0; phase < 3; ++phase) {
                for (int buffer = 0; buffer < 2; ++buffer) {
                    init_slot();
                    slot.active_mask = ActiveMask(SUBTASK_MASK_ALL);
                    payload.scalar_prefix_tagged_args = 56;
                    if (phase != 0) {
                        if (contract == 0) payload.scalar_prefix_tagged_args &= ~(member << 3);
                        if (contract == 1) payload.scalar_prefix = member;
                        if (contract == 2) payload.runtime_barrier_tail = member;
                        if (contract == 3) payload.runtime_compiled_poll = member;
                    }
                    const auto handle = context->prepare_subtask_to_core(
                        0, core_offset, slot, subslot, false, 0, false);
                    ASSERT_TRUE(handle.valid);
                    auto &ctx = context->payload_per_core_[core_offset][handle.reg_task_id & 1u].global_context;
                    EXPECT_EQ(context->payload_per_core_[core_offset][handle.reg_task_id & 1u].local_context.async_ctx.local_predecessor_pending, phase == 0 ? DISPATCH_ORDINARY_ENTRY : 0u);
                    if (phase != 0) {
                        EXPECT_EQ(ctx.predecessor_context, 0u);
                        EXPECT_EQ(ctx.wait_predecessor_fn, 0u);
                        EXPECT_EQ(ctx.scalar_overlap_trace, 0u);
                        ctx.predecessor_context = 64;
                        ctx.wait_predecessor_fn = 128;
                        ctx.scalar_overlap_trace = 192;
                    }
                }
            }
        }
    }
}
