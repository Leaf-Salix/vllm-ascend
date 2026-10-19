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

AICORE void disjoint_rows(GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1, GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v3 = 1024;
  // pto: %c0
  const int64_t v4 = 0;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c2
  const int64_t v6 = 2;
  // pto: %c64
  const int64_t v7 = 64;
  // pto: %a0
  const int64_t v8 = 0;
  // pto: %a65536
  const int64_t v9 = 65536;
  // pto: %zero
  const bfloat16_t v10 = 0.0f;
  // pto: %parent
  Tile<TileType::Vec, bfloat16_t, 64, 512, BLayout::RowMajor, 64, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v11;
  // pto: %parent
  uint64_t v12 = (uint64_t) v8;
  TASSIGN(v11, v12);
  // pto: %zero_row
  Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v13;
  // pto: %zero_row
  uint64_t v14 = (uint64_t) v9;
  TASSIGN(v13, v14);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TEXPANDS(v11, v10);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TEXPANDS(v13, v10);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v15 = v4; v15 < v7; v15 += v5) {
    // pto: %target
    ;
    Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16;
    // pto: %target
    ;
    uint64_t v17 = (uint64_t) ((int64_t) (uint64_t) v15 * (uint64_t) v3);
    TASSIGN(v16, v17);
    // pto: %parity, %take_dma
    ;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if ((int64_t) ((uint64_t) v15 % (uint64_t) v6) == v4) {
      TLOAD(v16, v1);
    } else {
      pipe_barrier(PIPE_V);
      TMOV(v16, v13);
    };
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v2, v11);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}