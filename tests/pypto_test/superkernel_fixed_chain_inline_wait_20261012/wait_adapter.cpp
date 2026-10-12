
#include "common/platform_config.h"
static __aicore__ __attribute__((always_inline)) inline
uint32_t pypto_boundary_dispatch_token() {
    uint32_t value;
    asm volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(value));
    return value;
}
static __aicore__ __attribute__((always_inline)) inline
uint32_t pypto_boundary_gate_token() {
    uint64_t value;
    asm volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(value));
    return static_cast<uint32_t>(value >> 32);
}
static __aicore__ __attribute__((always_inline)) inline
int pypto_inline_clean_boundary(__gm__ int64_t* args, __gm__ GlobalContext* ctx) {
    auto* previous = reinterpret_cast<__gm__ GlobalContext*>(ctx->predecessor_context);
    bool clean = (ctx->scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) != 0 &&
                 ctx->scalar_overlap_trace == 0;
    if (previous != nullptr) {
        const auto kind = static_cast<AsyncTailKind>(previous->async_tail_kind);
        clean = clean && previous->scalar_overlap_trace == 0 &&
                (kind == AsyncTailKind::RuntimeBarrierAll ||
                 kind == AsyncTailKind::RuntimeCompiledPollClean ||
                 kind == AsyncTailKind::RuntimeDirectPollClean);
    }
    if (!clean) {
        using WaitFn = int (*)(__gm__ int64_t*);
        return reinterpret_cast<WaitFn>(ctx->wait_predecessor_fn)(args);
    }
    // Identical ordering to the runtime callback. Only the indirect call and
    // legacy/diagnostic branches are absent from this clean tagged path.
    if (previous != nullptr) pipe_barrier(PIPE_ALL);
    const uint32_t predecessor_id = ctx->predecessor_reg_task_id;
    const uint32_t own_id = ctx->reg_task_id;
    dcci(reinterpret_cast<__gm__ char*>(ctx) + offsetof(GlobalContext, async_tail_kind),
         SINGLE_CACHE_LINE, CACHELINE_OUT);
    auto* own = reinterpret_cast<__gm__ DispatchPayload*>(
        reinterpret_cast<__gm__ char*>(ctx) - offsetof(DispatchPayload, global_context));
    const bool gated = own->src_payload != 0;
    if (previous != nullptr) set_cond(static_cast<uint32_t>(MAKE_FIN_VALUE(predecessor_id)));
    if (gated) {
        while (pypto_boundary_gate_token() != own_id) {
            if (pypto_boundary_dispatch_token() == AICORE_EXIT_SIGNAL) return 0;
        }
    }
    if (pypto_boundary_dispatch_token() == AICORE_EXIT_SIGNAL) return 0;
    set_cond(static_cast<uint32_t>(MAKE_ACK_VALUE(own_id)));
    return 1;
}

extern "C" {
static __aicore__ __attribute__((always_inline)) inline
int64_t pypto_wait_local_boundary(__gm__ int64_t* args, int64_t has_predecessor) {
    asm volatile("" : : : "memory");
    int64_t ready = 1;
    if (has_predecessor != 0) {
        auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
        ready = pypto_inline_clean_boundary(args, ctx);
    }
    // The Scalar prefix must not change state inherited by the predecessor.
    // PTOAS suppresses its normal prologue on marked functions; initialize the
    // vector/atomic modes only after the runtime completed the predecessor.
    if (ready) {
        set_atomic_none();
#if defined(__DAV_VEC__)
        set_mask_norm();
        set_vector_mask(-1, -1);
#endif
    }
    asm volatile("" : : : "memory");
    return ready;
}
static __aicore__ __attribute__((always_inline)) inline
int64_t pypto_wait_legacy_boundary(__gm__ int64_t* args) {
    auto* local = reinterpret_cast<__gm__ LocalContext*>(args[SPMD_LOCAL_CONTEXT_INDEX]);
    return pypto_wait_local_boundary(args, local->async_ctx.local_predecessor_pending);
}
static __aicore__ __attribute__((always_inline)) inline
int64_t pypto_wait_predecessor(__gm__ int64_t* args, int64_t dependency) {
    asm volatile("" : "+l"(dependency) : : "memory");
    return pypto_wait_legacy_boundary(args);
}
}
