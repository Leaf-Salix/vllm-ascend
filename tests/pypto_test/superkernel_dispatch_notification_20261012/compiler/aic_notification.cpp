static __aicore__ __attribute__((noinline)) void pypto_poll_local_tail(__gm__ int64_t* args) {
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    auto* notice = reinterpret_cast<__gm__ TaskBoundaryNotification*>(ctx->task_boundary_notification);
    auto* value = reinterpret_cast<volatile __gm__ uint32_t*>(notice->value);
    const uint32_t own = ctx->reg_task_id;
    // The high bit separates valid dispatch generations from cold/legacy
    // values. After ID reuse, retain the original read-and-invert protocol.
    uint32_t expected = own | uint32_t{0x80000000};
    if ((ctx->late_handoff_scratch & COMPILED_POLL_ID_REUSED) != 0) {
        dcci(notice->value, SINGLE_CACHE_LINE);
        expected = *value == 0 ? ~uint32_t{0} : 0;
    }
    auto* scratch = reinterpret_cast<__cbuf__ uint32_t*>(0);
    create_cbuf_matrix(scratch, int64_t{0x10002}, expected);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_M, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_MTE3);
    copy_cbuf_to_gm(reinterpret_cast<__gm__ uint32_t*>(notice->value), scratch, 0, 1, 2, 0, 0);
    const bool trace = ctx->scalar_overlap_trace != 0;
    if (trace) ctx->late_handoff_scratch |= uint64_t{1} << 32;
    for (;;) {
        uint32_t next;
        asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(next) : : "memory");
        if (next != own && next != 0x7ffffffdu) {
            return;
        }
        dcci(notice->value, SINGLE_CACHE_LINE);
        if (*value == expected) return;
    }
}
