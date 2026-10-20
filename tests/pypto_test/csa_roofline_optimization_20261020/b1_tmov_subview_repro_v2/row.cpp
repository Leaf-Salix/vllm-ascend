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

AICORE void assemble_row(int64_t v1) {
  using T = float;
  const int64_t v2 = 4096;
  // pto: %src
  const int64_t v3 = 0;
  // pto: %src
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v4;
  // pto: %src
  uint64_t v5 = (uint64_t) v3;
  TASSIGN(v4, v5);
  // pto: %parent
  uint64_t v6 = (uint64_t) v2;
  // pto: %view
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v7;
  // pto: %view
  uint64_t v8 = (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v1 * (uint64_t) v2) + (uint64_t) v2);
  TASSIGN(v7, v8);
  TMOV(v7, v4);
  return;
}