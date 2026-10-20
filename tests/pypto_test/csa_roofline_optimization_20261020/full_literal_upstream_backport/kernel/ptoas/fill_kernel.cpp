#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


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

AICORE void fill_kernel(__gm__ float* v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v2 = 0;
  // pto: %c1_index
  const int64_t v3 = 1;
  // pto: %c64_index
  const int64_t v4 = 64;
  // pto: %cst_3
  const float v5 = 0.0f;
  // pto: %out__ssa_v0_view
  const int64_t v6 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v7 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v8 = 1;
  // pto: %out__ssa_v0_view
  int64_t v9 = v3 * v4;
  // pto: %out__ssa_v0_view
  int64_t v10 = v8 * v9;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v11 = pto::Shape<1, 1, 1, -1, -1>(v6, v7, v8, v3, v4);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v12 = pto::Stride<-1, -1, -1, -1, -1>(v7 * v10, v10, v9, v4, v3);
  // pto: %out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v13 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v11, v12);
  // pto: %value__ssa_v0
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v14 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v3, v4);
  // pto: %value__ssa_v0
  uint64_t v15 = (uint64_t) v2;
  TASSIGN(v14, v15);
  TEXPANDS(v14, v5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %out__ssa_v0_pview
  __gm__ float* v16 = PTOAS__GLOBAL_TENSOR_DATA(v13);
  // pto: %out__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 64> v17 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %out__ssa_v0_pview
  pto::Stride<64, 64, 64, 64, 1> v18 = pto::Stride<64, 64, 64, 64, 1>();
  // pto: %out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v16, v17, v18);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v19, v14);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}