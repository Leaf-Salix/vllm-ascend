// PYPTO_TASK_BOUNDARY_RELAY_V1 late_producer MTE2_TO_V=7 V_TO_MTE3=7
static __aicore__ __attribute__((noinline)) void pypto_poll_local_tail(__gm__ int64_t* args) {
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    auto* notice = reinterpret_cast<__gm__ TaskBoundaryNotification*>(ctx->task_boundary_notification);
    auto* value = reinterpret_cast<volatile __gm__ uint64_t*>(notice->value);
    dcci(notice->value, SINGLE_CACHE_LINE);
    const uint64_t expected = *value == 0 ? ~uint64_t{0} : 0;
    auto* scratch = reinterpret_cast<__ubuf__ uint16_t*>(0);
    set_mask_count();
    set_vector_mask(0, 32);
    vector_dup(scratch, static_cast<uint16_t>(expected), 0, 1, 1, 8, 8);
    set_mask_norm();
    set_vector_mask(~uint64_t{0}, ~uint64_t{0});
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_MTE3);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ uint16_t*>(notice->value), scratch, 0, 1, 2, 0, 0);
    const uint32_t own = ctx->reg_task_id;
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
