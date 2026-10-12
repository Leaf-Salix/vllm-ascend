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

// PYPTO_TASK_BOUNDARY_EVENTS_V1 fenced_then_used MTE2=7 MTE1=7 M=7 FIX=7 V=6
#ifndef PYPTO_TASK_BOUNDARY_EVENTS_V1
#error allocated task boundary events require a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 paired_only
#ifndef PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1
#error blocking tail poll requires a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 single_pipe
#ifndef PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1
#error blocking tail poll requires a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 generated_tail
#ifndef PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1
#error blocking tail poll requires a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_EVENTS_V1 unmatched MTE2=7 MTE1=7 M=7 FIX=7 V=6
#ifndef PYPTO_TASK_BOUNDARY_EVENTS_V1
#error allocated task boundary events require a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_EVENTS_V1 nested_pair MTE2=7 MTE1=7 M=7 FIX=7 V=6
#ifndef PYPTO_TASK_BOUNDARY_EVENTS_V1
#error allocated task boundary events require a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1 dynamic_pair
#ifndef PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1
#error blocking tail poll requires a matching PyPTO wrapper
#endif
AICORE void drained_ids() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

AICORE void fenced_then_used() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  pipe_barrier(PIPE_ALL);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  #endif // __DAV_VEC__

  return;
}

AICORE void paired_only() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  #endif // __DAV_VEC__

  return;
}

AICORE void single_pipe() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  pipe_barrier(PIPE_V);
  #endif // __DAV_VEC__

  return;
}

AICORE void generated_tail() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void unmatched() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

AICORE void nested_pair(bool v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  if (v1) {
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    pipe_barrier(PIPE_ALL);
  }
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

AICORE void dynamic_pair(int64_t v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  event_t v2 = (event_t) v1;
  set_flag(PIPE_V, PIPE_MTE3, v2);
  event_t v3 = (event_t) v1;
  wait_flag(PIPE_V, PIPE_MTE3, v3);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}