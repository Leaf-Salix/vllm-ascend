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

extern "C" AICORE int64_t pypto_volatile_read_i64(__gm__ int64_t*, int64_t);
AICORE void volatile_scalar_call_probe(__gm__ int64_t* v1, __gm__ int64_t* v2) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %zero
  const int64_t v3 = 0;
  // pto: %value
  int64_t v4 = pypto_volatile_read_i64(v1, v3);
  (v2)[v3] = v4;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_CUBE__

  return;
}