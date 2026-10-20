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

AICORE bool mixed_bool(bool v1, bool v2, bool v3, bool v4, bool v5) {
  using T = float;
  // pto: %cd
  bool v6 = v3 | v4;
  // pto: %ab, %x, %y
  return (v1 & v2 & v6) & v5;
}

AICORE int32_t mixed_bits(int32_t v1, int32_t v2, int32_t v3, int32_t v4) {
  using T = float;
  // pto: %bc
  uint32_t v5 = (uint32_t) v2 | (uint32_t) v3;
  // pto: %bc, %bcd
  uint32_t v6 = (uint32_t) ((int32_t) v5) ^ (uint32_t) v4;
  // pto: %result, %bcd
  return (int32_t) ((uint32_t) v1 & (uint32_t) ((int32_t) v6));
}