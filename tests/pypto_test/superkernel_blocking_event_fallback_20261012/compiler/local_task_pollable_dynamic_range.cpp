// End of FileCheck-validated generated output.
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

// PYPTO_TASK_BOUNDARY_EVENTS_V1 upper_half MTE2=3 MTE1=7 M=7 FIX=7 V=7
#ifndef PYPTO_TASK_BOUNDARY_EVENTS_V1
#error allocated task boundary events require a matching PyPTO wrapper
#endif
// PYPTO_TASK_BOUNDARY_EVENTS_V1 either_edge MTE2=6 MTE1=7 M=7 FIX=7 V=7
#ifndef PYPTO_TASK_BOUNDARY_EVENTS_V1
#error allocated task boundary events require a matching PyPTO wrapper
#endif
AICORE void upper_half(int64_t v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %mask
  const int64_t v2 = 3;
  // pto: %four
  const int64_t v3 = 4;
  // pto: %low, %id
  int64_t v4 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v1 & (uint64_t) v2) + (uint64_t) v3);
  event_t v5 = (event_t) v4;
  set_flag(PIPE_MTE2, PIPE_MTE3, v5);
  event_t v6 = (event_t) v4;
  wait_flag(PIPE_MTE2, PIPE_MTE3, v6);
  #endif // __DAV_VEC__

  return;
}

AICORE void either_edge(bool v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %zero
  const int64_t v2 = 0;
  // pto: %seven
  const int64_t v3 = 7;
  // pto: %id
  int64_t v4 = v1 ? v2 : v3;
  event_t v5 = (event_t) v4;
  set_flag(PIPE_MTE2, PIPE_MTE3, v5);
  event_t v6 = (event_t) v4;
  wait_flag(PIPE_MTE2, PIPE_MTE3, v6);
  #endif // __DAV_VEC__

  return;
}
