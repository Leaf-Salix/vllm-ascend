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

AICORE void same_row(GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1, GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0
  const int64_t v3 = 0;
  // pto: %c1
  const int64_t v4 = 1;
  // pto: %c2
  const int64_t v5 = 2;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %a0
  const int64_t v7 = 0;
  // pto: %a65536
  const int64_t v8 = 65536;
  // pto: %zero
  const bfloat16_t v9 = 0.0f;
  // pto: %parent
  Tile<TileType::Vec, bfloat16_t, 64, 512, BLayout::RowMajor, 64, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v10;
  // pto: %parent
  uint64_t v11 = (uint64_t) v7;
  TASSIGN(v10, v11);
  // pto: %zero_row
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v12;
  // pto: %zero_row
  uint64_t v13 = (uint64_t) v8;
  TASSIGN(v12, v13);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TEXPANDS(v10, v9);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TEXPANDS(v12, v9);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v14 = v3; v14 < v6; v14 += v4) {
    // pto: %target
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v15;
    // pto: %target
    ;
    uint64_t v16 = (uint64_t) v7;
    TASSIGN(v15, v16);
    // pto: %parity, %take_dma
    ;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if ((int64_t) ((uint64_t) v14 % (uint64_t) v5) == v3) {
      TLOAD(v15, v1);
    } else {
      pipe_barrier(PIPE_V);
      TMOV(v15, v12);
    };
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v2, v10);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}