#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
  kAsyncAivEvent7 = 2,
  kAsyncAicEvent7 = 3,
  kAsyncAivAllPipesEvent7 = 4,
  kAsyncAicAllPipesEvent7 = 5,
  kRuntimeBarrierAll = 6,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kRuntimeBarrierAll:
    // The dynamic caller executes the ordinary PIPE_ALL before publishing FIN
    // or before a same-core successor may touch pipeline resources.
    break;
  // Every event targets Scalar, so internal uses of the same pair/ID have
  // already been consumed before Scalar can issue these tail markers.
  // The caller must join all listed engines before reusing this core.
  case PTOAutoSyncTailMode::kAsyncAicAllPipesEvent7:
    set_flag(PIPE_M, PIPE_S, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAivAllPipesEvent7:
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAicEvent7:
    // The caller drains Cube and FIX independently. Join every upstream
    // matrix pipeline without making Scalar wait for device completion.
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    set_flag(PIPE_M, PIPE_S, EVENT_ID7);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAivEvent7:
    // Join MTE2/V/MTE3 without a Scalar wait. The caller owns the final
    // MTE3->S event and must consume it before publishing completion.
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // Diagnostic completion marker: the caller measures Vector separately
    // from MTE3 so a trailing store cannot be mistaken for compute overlap.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

// PYPTO_TASK_BOUNDARY_RELAY_V1 vector_relay MTE2_TO_V=6 V_TO_MTE3=7
#ifndef PYPTO_TASK_BOUNDARY_RELAY_V1
#error relayed tail notification requires a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_RELAY_V1 cube_relay MTE2_TO_MTE1=6 MTE1_TO_M=5 M_TO_FIX=4 FIX_TO_MTE3=3
#ifndef PYPTO_TASK_BOUNDARY_RELAY_V1
#error relayed tail notification requires a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 no_free_route
#ifndef PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1
#error blocking tail poll requires a matching PyPTO wrapper
#endif
AICORE void vector_relay(int64_t v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  event_t v2 = (event_t) v1;
  set_flag(PIPE_MTE2, PIPE_MTE3, v2);
  event_t v3 = (event_t) v1;
  wait_flag(PIPE_MTE2, PIPE_MTE3, v3);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  #endif // __DAV_VEC__

  return;
}

AICORE void cube_relay(int64_t v1) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v2 = 1;
  const int64_t v3 = 0;
  const int64_t v4 = 6;
  const int64_t v5 = 5;
  const int64_t v6 = 4;
  event_t v7 = (event_t) v1;
  set_flag(PIPE_MTE2, PIPE_MTE3, v7);
  event_t v8 = (event_t) v1;
  wait_flag(PIPE_MTE2, PIPE_MTE3, v8);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
  // pto: %bit
  int64_t v9 = (int64_t) ((uint64_t) v1 & (uint64_t) v2);
  // pto: %cond
  bool v10 = v9 == v3;
  // pto: %f
  int64_t v11 = (int64_t) ((uint64_t) v9 + (uint64_t) v4);
  // pto: %m
  int64_t v12 = v10 ? v5 : v11;
  int64_t v13 = v10 ? v6 : v11;
  event_t v14 = (event_t) v12;
  set_flag(PIPE_M, PIPE_FIX, v14);
  event_t v15 = (event_t) v12;
  wait_flag(PIPE_M, PIPE_FIX, v15);
  event_t v16 = (event_t) v13;
  set_flag(PIPE_FIX, PIPE_MTE3, v16);
  event_t v17 = (event_t) v13;
  wait_flag(PIPE_FIX, PIPE_MTE3, v17);
  #endif // __DAV_CUBE__

  return;
}

AICORE void no_free_route(int64_t v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  event_t v2 = (event_t) v1;
  set_flag(PIPE_V, PIPE_MTE3, v2);
  event_t v3 = (event_t) v1;
  wait_flag(PIPE_V, PIPE_MTE3, v3);
  #endif // __DAV_VEC__

  return;
}
