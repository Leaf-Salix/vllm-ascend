/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicore/chip_swimlane_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"  // Register-based communication
#include "dispatch_payload.h"
#include "runtime.h"
#include "task_interface/tmr_kernel_context.h"
#include "task_interface/tmr_kernel_control.h"

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

__aicore__ __attribute__((always_inline)) static void flush_overlap_context(__gm__ GlobalContext *ctx) {
    // The CPU reuses this dual-buffer slot after FIN. Leave no dirty AICore
    // control line that a later invalidation could write over its next header.
    dcci(
        reinterpret_cast<__gm__ char *>(ctx) + offsetof(GlobalContext, async_tail_kind),
        SINGLE_CACHE_LINE, CACHELINE_OUT
    );
}

__aicore__ __attribute__((always_inline)) static void drain_async_tail(
    __gm__ GlobalContext *ctx, AsyncTailKind kind
) {
    const uint64_t trace_addr = ctx->scalar_overlap_trace;
    uint64_t wait_started_at;
    uint64_t compute_completed_at;
    const bool runtime_tail = kind == AsyncTailKind::RuntimeBarrierAll ||
                              kind == AsyncTailKind::RuntimePollable || kind == AsyncTailKind::RuntimeCompiledPoll;
    if (runtime_tail) {
        wait_started_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        pipe_barrier(PIPE_ALL);
        compute_completed_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
    } else if (kind == AsyncTailKind::AicJoined || kind == AsyncTailKind::AicAllPipes) {
        wait_started_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        wait_flag(PIPE_M, PIPE_S, EVENT_ID7);
        compute_completed_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        if (kind == AsyncTailKind::AicAllPipes) {
            wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        }
    } else {
        wait_started_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        compute_completed_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        if (kind == AsyncTailKind::AivAllPipes) {
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        }
    }
    if (trace_addr != 0) {
        const uint64_t completed_at = get_sys_cnt_aicore();
        auto *trace = reinterpret_cast<__gm__ uint64_t *>(trace_addr);
        trace[0] = get_raw_core_id();
        trace[1] = ctx->scalar_overlap_start;
        trace[2] = ctx->scalar_overlap_returned;
        trace[3] = completed_at;
        trace[4] = 7;
        if (kind == AsyncTailKind::RuntimePollable || kind == AsyncTailKind::RuntimeCompiledPoll) {
            trace[4] |= (ctx->late_handoff_scratch >> 32) << 8;
        }
        trace[5] = wait_started_at;
        trace[6] = completed_at - wait_started_at;
        trace[7] = compute_completed_at;
        dcci(trace, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    if (!runtime_tail) {
        ctx->async_tail_kind = 0;
        flush_overlap_context(ctx);
    } else if (trace_addr != 0 || kind != AsyncTailKind::RuntimeBarrierAll) {
        flush_overlap_context(ctx);
    }
}

__aicore__ __attribute__((always_inline)) static void drain_async_tail(__gm__ GlobalContext *ctx) {
    drain_async_tail(ctx, static_cast<AsyncTailKind>(ctx->async_tail_kind));
}

// Called after a checked Scalar prefix to consume its local tail, if any, and
// its input/cohort gate. Delayed ACK preserves the scheduler's occupied slot;
// a pending ACK is evidence that the local predecessor has finished.
__aicore__ static int wait_scalar_prefix_predecessor(__gm__ int64_t *args) {
    auto *ctx = reinterpret_cast<__gm__ GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    auto *previous = reinterpret_cast<__gm__ GlobalContext *>(ctx->predecessor_context);
    if (previous != nullptr) drain_async_tail(previous);
    const uint32_t predecessor_id = ctx->predecessor_reg_task_id;
    const uint32_t own_id = ctx->reg_task_id;
    ctx->predecessor_context = 0;
    ctx->wait_predecessor_fn = 0;
    // Only a deferred boundary dirties these fields. Flush before either task
    // can publish FIN; ordinary runtime-owned tails keep the control line clean.
    flush_overlap_context(ctx);
    auto *own = reinterpret_cast<__gm__ DispatchPayload *>(
        reinterpret_cast<__gm__ char *>(ctx) - offsetof(DispatchPayload, global_context));
    if ((ctx->scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) == 0) {
        // Legacy entries still consume a GM flag. Register-argument entries
        // never dirty the hot header, so they need no additional writeback.
        own->local_context.async_ctx.local_predecessor_pending = 0;
        dcci(own, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    const bool gated = own->src_payload != 0;
    if (previous != nullptr) write_reg(RegId::COND, MAKE_FIN_VALUE(predecessor_id));
    // Gated consumers still wait for every input producer. An already-ready
    // successor has no doorbell to await: its inputs preceded normal dispatch.
    if (gated) {
        while (read_dmb_high32() != own_id) {
            if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) return 0;
            SPIN_WAIT_HINT();
        }
    }
    if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) return 0;
    write_reg(RegId::COND, MAKE_ACK_VALUE(own_id));
    return 1;
}

__aicore__ static uint32_t handoff_scalar_prefix(
    __gm__ DispatchPayload *base, __gm__ DispatchPayload *current, uint32_t task_id
) {
    const uint32_t next_id = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    if (next_id == task_id || next_id == AICPU_IDLE_TASK_ID || next_id == AICORE_EXIT_SIGNAL) return AICPU_IDLE_TASK_ID;
    auto *next = base + (next_id & 1u);
    dcci(next, ENTIRE_DATA_CACHE);
    if ((next->global_context.scalar_prefix & SCALAR_PREFIX_CAN_HANDOFF) == 0 ||
        next->global_context.predecessor_reg_task_id != task_id) return AICPU_IDLE_TASK_ID;
    next->global_context.predecessor_context = reinterpret_cast<uint64_t>(&current->global_context);
    next->global_context.wait_predecessor_fn = reinterpret_cast<uint64_t>(&wait_scalar_prefix_predecessor);
    if ((next->global_context.scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) == 0) {
        next->local_context.async_ctx.local_predecessor_pending = 1;
    }
    return next_id;
}

// The task image owns its reserved scratch and DMA instruction sequence.
// Re-enter only its private tail callback after the original probe misses;
// argument bit 1 bypasses the entire operator body. No FIN/ACK is emitted here.
__aicore__ __attribute__((noinline)) static uint32_t poll_late_handoff(
    __gm__ DispatchPayload *base, __gm__ DispatchPayload *current, uint32_t task_id
) {
    const uint32_t dispatched = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    if (dispatched != task_id && dispatched != AICPU_IDLE_TASK_ID) {
        return handoff_scalar_prefix(base, current, task_id);
    }
    auto kernel = reinterpret_cast<UnifiedKernelFunc>(current->function_bin_addr);
    auto *callback_args = reinterpret_cast<__gm__ int64_t *>(reinterpret_cast<uint64_t>(current->args) | 2u);
    kernel(callback_args);
    const uint32_t candidate = handoff_scalar_prefix(base, current, task_id);
    if (candidate != AICPU_IDLE_TASK_ID && current->global_context.scalar_overlap_trace != 0) {
        current->global_context.late_handoff_scratch |= uint64_t{2} << 32;
    }
    return candidate;
}

/**
 * Execute task from DispatchPayload.
 *
 * Reads function_bin_addr and args from the dispatch payload.
 *
 * @param payload Pointer to DispatchPayload in global memory
 */
__aicore__ __attribute__((always_inline)) static void execute_task(
    __gm__ DispatchPayload *payload, __gm__ int64_t *args
) {
    if (payload == nullptr || payload->function_bin_addr == 0) {
        return;
    }

    auto kernel = reinterpret_cast<UnifiedKernelFunc>(payload->function_bin_addr);
    kernel(args);
    if (payload->global_context.scalar_overlap_trace != 0) {
        payload->global_context.scalar_overlap_returned = get_sys_cnt_aicore();
    }
    OUT_OF_ORDER_STORE_BARRIER();
}

using simpler::tmr::TmrCoreReport;
using simpler::tmr::TmrKernelContextDescriptor;
using simpler::tmr::TmrLaunchControl;

__aicore__ static void execute_dispatch_loop(__gm__ Handshake *my_hank) {
    dcci(my_hank, SINGLE_CACHE_LINE);
    __gm__ DispatchPayload *payload = reinterpret_cast<__gm__ DispatchPayload *>(my_hank->task);

    uint32_t enable_profiling_flag = get_aicore_profiling_flag();
    bool chip_swimlane_enabled = SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    bool dump_args_enabled = SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS);
    bool pmu_enabled = SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_PMU);

    // This executor chooses first-dispatch lazy resolution. The rotation
    // channel is safe to resolve after the wrapper observes window-open.
    __gm__ ChipSwimlaneActiveHead *chip_swimlane_head = nullptr;
    // cached_buf_seq must start != AICPU's initial head.current_buf_seq (0)
    // so the first reservation observes a mismatch and loads the buffer ptr.
    ChipSwimlaneAicoreLocalState chip_swimlane_local = {nullptr, UINT32_MAX, 0};

    // Register encoding: AICPU_IDLE_TASK_ID=idle, task_id=task, AICORE_EXIT_SIGNAL=exit
    uint32_t reg_val = AICPU_IDLE_TASK_ID;
    uint32_t last_reg_val = AICPU_IDLE_TASK_ID;
    uint32_t prepared_prefix_task_id = AICPU_IDLE_TASK_ID;
    __gm__ GlobalContext *prepared_predecessor = nullptr;
    bool exiting = false;

    while (true) {
        reg_val = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (reg_val == AICORE_EXIT_SIGNAL) {
            // Teardown may race the handoff before the successor enters its
            // boundary. The previous task still owns unconsumed tail events.
            if (prepared_predecessor != nullptr) drain_async_tail(prepared_predecessor);
            // Signal exit acknowledgment to AICPU
            write_reg(RegId::COND, AICORE_EXITED_VALUE);
            break;
        }

        // Execute task if new (reg_val encoding: AICPU_IDLE_TASK_ID=idle, task_id=task)
        if (reg_val == AICPU_IDLE_TASK_ID || reg_val == last_reg_val) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            // receive_time = task pickup: DATA_MAIN_BASE returned a new task_id.
            // Paired with start_time (captured after the per-task dcci + ack) it
            // lets DFX split head_OH into the AICPU→AICore-ready propagation
            // (dispatch_ts → receive_time) and the AICore-local prep
            // (receive_time → start_time). Stored as a 32-bit delta
            // `start_time - receive_time`.
            //
            // Common path (src_payload == 0): the new task_id is itself the ready
            // signal, so receive_time is the true ready moment. Early-dispatch path
            // (src_payload != 0): receive_time stays at pickup — before the
            // doorbell wait — so it precedes the producer's end_time; the host
            // folds it to start_time for those tasks (detected when receive
            // precedes the producer task's end_time).
            uint64_t receive_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

            uint32_t task_id = reg_val;  // Decode: register holds task_id directly

            if (chip_swimlane_enabled && chip_swimlane_head == nullptr) {
                chip_swimlane_head = get_chip_swimlane_aicore_head();
            }

            // Select dual-buffer slot: same bit as AICPU used when writing payload
            __gm__ DispatchPayload *exec_payload = payload + (task_id & 1u);

            // handoff_scalar_prefix already invalidated this exact generation.
            // Its ACK is withheld, so AICPU cannot recycle the slot meanwhile.
            // Do not flush/reload the entire cache a second time on the handoff.
            if (task_id != prepared_prefix_task_id) dcci(exec_payload, ENTIRE_DATA_CACHE);
            prepared_prefix_task_id = AICPU_IDLE_TASK_ID;
            prepared_predecessor = nullptr;
            bool scalar_prefix_handoff = exec_payload->global_context.predecessor_context != 0;

            // Early-dispatch gate. A gated task was staged on this core before its
            // dependencies resolved; wait until AICPU rings the doorbell
            // (DATA_MAIN_BASE high 32 == task_id) before executing. The ACK is
            // deferred until AFTER the gate so the scheduler keeps the core
            // off-limits (pending_occupied stays set, no ACK->pending_freed) while
            // the task is gated — preventing a real task from being dual-issued
            // behind it. The kernel's own input dcci runs inside execute_task()
            // below — strictly AFTER this gate — so predecessor outputs are visible.
            // src_payload == 0 (the common ready path) skips this; a non-zero
            // src_payload is both the gate flag and the source TaskPayload.
            if (exec_payload->src_payload != 0) {
                // AICPU staged only src_payload, not the arg vector. Prepare
                // args here for both ordinary gates and local-tail handoffs;
                // in the latter case this work can overlap the predecessor's
                // pipeline. The cache invalidation above (or in handoff) makes
                // the source metadata coherent before any kernel prefix runs.
                // Context arguments remain the AICPU's per-buffer constants.
                __gm__ char *src = reinterpret_cast<__gm__ char *>(exec_payload->src_payload);
                int32_t tensor_count = *reinterpret_cast<__gm__ int32_t *>(src + TASKPAYLOAD_TENSOR_COUNT_OFFSET);
                int32_t scalar_count = *reinterpret_cast<__gm__ int32_t *>(src + TASKPAYLOAD_SCALAR_COUNT_OFFSET);
                __gm__ uint64_t *src_scalars = reinterpret_cast<__gm__ uint64_t *>(src + TASKPAYLOAD_SCALARS_OFFSET);
                int n = 0;
                for (int32_t i = 0; i < tensor_count; i++) {
                    exec_payload->args[n++] =
                        reinterpret_cast<uint64_t>(src + TASKPAYLOAD_TENSORS_OFFSET + i * TASKPAYLOAD_TENSOR_STRIDE);
                }
                for (int32_t i = 0; i < scalar_count; i++) {
                    exec_payload->args[n++] = src_scalars[i];
                }
                OUT_OF_ORDER_STORE_BARRIER();
                // An idle core can run the same checked prefix without a local
                // predecessor. The callback retains the full-fanin/cohort gate
                // and delayed ACK, so no pending slot is released prematurely.
                if (!scalar_prefix_handoff && !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&
                    (exec_payload->global_context.scalar_prefix & SCALAR_PREFIX_CAN_EARLY_START) != 0) {
                    exec_payload->global_context.wait_predecessor_fn =
                        reinterpret_cast<uint64_t>(&wait_scalar_prefix_predecessor);
                    if ((exec_payload->global_context.scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) == 0) {
                        exec_payload->local_context.async_ctx.local_predecessor_pending = 1;
                    }
                    scalar_prefix_handoff = true;
                }
                while (!scalar_prefix_handoff) {
                    // Honor teardown: shutdown overwrites the low half with EXIT.
                    // Check it on the doorbell-match iteration too, so an EXIT that
                    // races in right after the matching doorbell still wins over
                    // executing the gated task.
                    if (read_dmb_high32() == task_id) {
                        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
                            exiting = true;
                        }
                        break;
                    }
                    if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
                        exiting = true;
                        break;
                    }
                    SPIN_WAIT_HINT();
                }
                if (exiting) {
                    write_reg(RegId::COND, AICORE_EXITED_VALUE);
                    break;
                }
            }

            // Bind this task to the currently-published buffer generation
            // before ACK makes progress visible to AICPU.
            __gm__ ChipSwimlaneAicoreTaskRecord *chip_swimlane_record = nullptr;
            if (chip_swimlane_enabled) {
                chip_swimlane_record =
                    chip_swimlane_aicore_reserve_task_record(chip_swimlane_head, &chip_swimlane_local);
            }

            auto *call_args = reinterpret_cast<__gm__ int64_t *>(exec_payload->args);
            if (!scalar_prefix_handoff) {
                write_reg(RegId::COND, MAKE_ACK_VALUE(task_id));
            } else {
                // Reuse the existing ACK/handoff branch. An ordinary task
                // performs no ABI lookup. Only a deferred boundary may tag the
                // aligned argument pointer, which the generated wrapper
                // decodes before any dereference. Keep this lookup volatile
                // so the compiler cannot speculate it onto the normal path.
                auto *flags = reinterpret_cast<volatile __gm__ uint32_t *>(
                    &exec_payload->global_context.scalar_prefix);
                if ((*flags & SCALAR_PREFIX_TAGGED_ARGS) != 0) {
                    call_args = reinterpret_cast<__gm__ int64_t *>(reinterpret_cast<uint64_t>(call_args) | 1u);
                }
            }

            // PMU window brackets kernel execution.
            if (pmu_enabled) {
                pmu_aicore_begin();
            }

            uint64_t start_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

            execute_task(exec_payload, call_args);

            const auto tail_kind = static_cast<AsyncTailKind>(exec_payload->global_context.async_tail_kind);
            if (tail_kind != AsyncTailKind::None) {
                // Prototype observation uses explicit trace tensors; existing
                // PMU/swimlane ownership remains serial until separately extended.
                if (!chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled) {
                    uint32_t next_id = handoff_scalar_prefix(payload, exec_payload, task_id);
                    if (next_id == AICPU_IDLE_TASK_ID && tail_kind == AsyncTailKind::RuntimeCompiledPoll) {
                        next_id = poll_late_handoff(payload, exec_payload, task_id);
                    }
                    if (next_id != AICPU_IDLE_TASK_ID) {
                        prepared_prefix_task_id = next_id;
                        prepared_predecessor = &exec_payload->global_context;
                        last_reg_val = reg_val;
                        continue;  // successor boundary will publish the real A FIN
                    }
                }
                drain_async_tail(&exec_payload->global_context, tail_kind);
            } else if (scalar_prefix_handoff) {
                // The local Wait callback already cleared and flushed its
                // fields. Only diagnostics can dirty this line after it returns.
                // Keep the conservative writeback for legacy prefix entries.
                if (exec_payload->global_context.scalar_overlap_trace != 0 ||
                    (exec_payload->global_context.scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) == 0) {
                    flush_overlap_context(&exec_payload->global_context);
                }
            }

            // Keep start_time -> end_time scoped to AICore execution.
            uint64_t end_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(task_id));

            if (pmu_enabled) {
                pmu_aicore_end();
            }

            if (dump_args_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            // Two identity fields go into the record (different roles):
            //   - task_token_raw (ring/local) is pulled from the dispatch
            //     payload's LocalContext.async_ctx — already in AICore cache
            //     from the just-completed task, no extra GM load. Host uses
            //     it as the canonical task identity for JSON output / ring
            //     decoding.
            //   - reg_task_id is `task_id` (= reg_val, the per-core dispatch
            //     token AICore just read from DATA_MAIN_BASE). Per-dispatch
            //     unique within this core; host uses it as the join key
            //     against the AICPU record stream. Required for correctness
            //     under SPMD (block_num > num_cores) and MIX cluster spread,
            //     where multiple dispatches of the same task share the same
            //     task_token_raw.
            if (chip_swimlane_enabled) {
                uint64_t task_token_raw = exec_payload->local_context.async_ctx.task_token.raw;
                chip_swimlane_aicore_commit_task_record(
                    chip_swimlane_record, task_token_raw, task_id, receive_time, start_time, end_time
                );
            }
        }
    }
}

__aicore__ static void
execute_worker(__gm__ Runtime *runtime, __gm__ Handshake *my_hank, int block_idx, CoreType core_type) {
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Each launch resets this SPR; the AICPU publishes task before opening it.
    while (read_reg(RegId::DATA_MAIN_BASE) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_IDLE_VALUE);
    execute_dispatch_loop(my_hank);
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
    wait_for_post_close_release(&runtime->dev.teardown_gates[block_idx].post_close_release);
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    execute_worker(runtime, &runtime->dev.workers[block_idx], block_idx, core_type);
}

__aicore__ __attribute__((weak)) void aicore_execute_kernel(
    __gm__ Runtime *runtime, __gm__ const TmrKernelContextDescriptor *context, int block_idx, CoreType core_type
) {
    if (block_idx < 0 || block_idx >= context->worker_count || context->worker_count > RUNTIME_MAX_WORKER ||
        context->reports_bytes != static_cast<uint64_t>(context->worker_count) * sizeof(TmrCoreReport) ||
        context->reports_address == 0 || context->reports_address > UINT64_MAX - context->reports_bytes ||
        context->reports_address % alignof(TmrCoreReport) != 0)
        return;
    auto *reports = reinterpret_cast<__gm__ Handshake *>(context->reports_address);
    execute_worker(runtime, &reports[block_idx], block_idx, core_type);
}
