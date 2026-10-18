#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
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

AICORE void shift_prev(__gm__ int64_t* v1, int64_t v2) {
  const int64_t v3 = 0;
  const int64_t v4 = 1;
  const int64_t v5 = 48;
  bool v6;
  int64_t v7;
  int64_t v8;
  int64_t v9;
  int64_t v10;
  int64_t v11;
  int64_t v12;
  using T = float;
  // pto: %zero
  // pto: %one
  // pto: %step
  v10 = v3;
  v11 = v3;
  v12 = v3;
  goto label2;
label2:
  // pto: %keep
  v6 = false;
  v6 = v10 < v2;
  if (v6) {
    goto label3;
  } else {
    goto label4;
  }
label3:
  // pto: %next_i
  // pto: %next_current
  ;
  v7 = (int64_t) ((uint64_t) v10 + (uint64_t) v4);
  ;
  v8 = (int64_t) ((uint64_t) v11 + (uint64_t) v5);
  ;
  v9 = v11;
  v10 = v7;
  v11 = v8;
  v12 = v9;
  goto label2;
label4:
  (v1)[v3] = v11;
  (v1)[v4] = v12;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  return;
}

AICORE void rotate_state(__gm__ int64_t* v1, int64_t v2) {
  const int64_t v3 = 0;
  const int64_t v4 = 1;
  const int64_t v5 = 2;
  const int64_t v6 = 3;
  bool v7;
  int64_t v8;
  int64_t v9;
  int64_t v10;
  int64_t v11;
  int64_t v12;
  int64_t v13;
  int64_t v14;
  int64_t v15;
  using T = float;
  // pto: %zero
  // pto: %one
  // pto: %two
  // pto: %three
  v12 = v3;
  v13 = v4;
  v14 = v5;
  v15 = v6;
  goto label2;
label2:
  // pto: %keep
  v7 = false;
  v7 = v12 < v2;
  if (v7) {
    goto label3;
  } else {
    goto label4;
  }
label3:
  // pto: %next_i
  ;
  v8 = (int64_t) ((uint64_t) v12 + (uint64_t) v4);
  ;
  v9 = v14;
  ;
  v10 = v15;
  ;
  v11 = v13;
  v12 = v8;
  v13 = v9;
  v14 = v10;
  v15 = v11;
  goto label2;
label4:
  (v1)[v3] = v13;
  (v1)[v4] = v14;
  (v1)[v5] = v15;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  return;
}

AICORE void conditional_swap(__gm__ int64_t* v1, int64_t v2, bool v3) {
  const int64_t v4 = 0;
  const int64_t v5 = 1;
  const int64_t v6 = 11;
  const int64_t v7 = 22;
  bool v8;
  int64_t v9;
  bool v10;
  int64_t v11;
  int64_t v12;
  int64_t v13;
  int64_t v14;
  int64_t v15;
  int64_t v16;
  int64_t v17;
  int64_t v18;
  int64_t v19;
  using T = float;
  // pto: %zero
  // pto: %one
  // pto: %eleven
  // pto: %twenty_two
  v17 = v4;
  v18 = v6;
  v19 = v7;
  goto label2;
label2:
  // pto: %keep
  v8 = false;
  v8 = v17 < v2;
  if (v8) {
    goto label3;
  } else {
    goto label4;
  }
label3:
  // pto: %next_i
  v9 = (int64_t) ((uint64_t) v17 + (uint64_t) v5);
  v10 = false;
  v10 = v3;
  ;
  v11 = v9;
  ;
  v12 = v19;
  ;
  v13 = v18;
  ;
  v14 = v9;
  ;
  v15 = v18;
  ;
  v16 = v19;
  if (v10) {
    v17 = v11;
    v18 = v12;
    v19 = v13;
    goto label2;
  } else {
    v17 = v14;
    v18 = v15;
    v19 = v16;
    goto label2;
  }
label4:
  (v1)[v4] = v18;
  (v1)[v5] = v19;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  return;
}