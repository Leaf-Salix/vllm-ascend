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

AICORE void casts(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ float* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const float v4 = -1.0f;
  const int32_t v5 = 0;
  const int32_t v6 = -1;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c4_index
  const int64_t v8 = 4;
  // pto: %c0_index
  const int64_t v9 = 0;
  // pto: %c2_index
  const int64_t v10 = 2;
  // pto: %c3_index
  const int64_t v11 = 3;
  // pto: %c5_index
  const int64_t v12 = 5;
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  // pto: %value__tile
  int32_t v13 = (v1)[v9];
  // pto: %0, %1
  bool v14 = (int64_t) v13 > v9;
  (v2)[v9] = v6;
  (v2)[v7] = v5;
  (v2)[v10] = v6;
  // pto: %11
  const int32_t v15 = 0;
  // pto: %11
  int32_t v16 = v15 - (int32_t) v14;
  (v2)[v11] = v16;
  (v2)[v8] = v6;
  // pto: %4, %13
  int32_t v17 = (int32_t) ((int64_t) v14);
  (v2)[v12] = v17;
  (v3)[v9] = v4;
  // pto: %16
  float v18 = (float) v14;
  (v3)[v7] = v18;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}