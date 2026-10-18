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

AICORE void tile_kernel(__gm__ float* v1, __gm__ float* v2, int64_t v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c64_index
  const int64_t v5 = 64;
  // pto: %c1_index
  const int64_t v6 = 1;
  // pto: %c8_index
  const int64_t v7 = 8;
  // pto: %c0_index
  const int64_t v8 = 0;
  // pto: %source__ssa_v0_view
  const int64_t v9 = 1;
  // pto: %source__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %source__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %source__ssa_v0_view
  int64_t v12 = (int64_t) v3;
  // pto: %source__ssa_v0_view
  int64_t v13 = v12 * v5;
  // pto: %source__ssa_v0_view
  int64_t v14 = v11 * v13;
  // pto: %source__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v9, v10, v11, v12, v5);
  // pto: %source__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v14, v14, v13, v5, v6);
  // pto: %source__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v17 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v15, v16);
  // pto: %output__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %output__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %output__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %output__ssa_v0_view
  int64_t v21 = (int64_t) v3;
  // pto: %output__ssa_v0_view
  int64_t v22 = v21 * v5;
  // pto: %output__ssa_v0_view
  int64_t v23 = v20 * v22;
  // pto: %output__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v18, v19, v20, v21, v5);
  // pto: %output__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v23, v23, v22, v5, v6);
  // pto: %output__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v24, v25);
  // pto: %0
  int64_t v27 = v3 < v7 ? v3 : v7;
  // pto: %tile__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v5);
  // pto: %tile__ssa_v0
  uint64_t v29 = (uint64_t) v4;
  TASSIGN(v28, v29);
  // pto: %source__ssa_v0_pview
  const int64_t v30 = 0;
  // pto: %source__ssa_v0_pview
  __gm__ float* v31 = PTOAS__GLOBAL_TENSOR_DATA(v17);
  // pto: %source__ssa_v0_pview
  const int64_t v32 = 1;
  // pto: %source__ssa_v0_pview
  const int64_t v33 = 1;
  // pto: %source__ssa_v0_pview
  const int64_t v34 = 1;
  // pto: %source__ssa_v0_pview
  int64_t v35 = v27 * v5;
  // pto: %source__ssa_v0_pview
  int64_t v36 = v34 * v35;
  // pto: %source__ssa_v0_pview
  pto::Shape<1, 1, 1, -1, 64> v37 = pto::Shape<1, 1, 1, -1, 64>(v32, v33, v34, v27, v5);
  // pto: %source__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v36, v36, v35, v5, v6);
  // pto: %source__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v31 + (v30 + v8 * v5 + v8 * v6), v37, v38);
  TLOAD(v28, v39);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  // pto: %same_tile__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v5);
  // pto: %same_tile__ssa_v0
  uint64_t v41 = (uint64_t) v4;
  TASSIGN(v40, v41);
  // pto: %output__ssa_v0_pview
  const int64_t v42 = 0;
  // pto: %output__ssa_v0_pview
  __gm__ float* v43 = PTOAS__GLOBAL_TENSOR_DATA(v26);
  // pto: %output__ssa_v0_pview
  const int64_t v44 = 1;
  // pto: %output__ssa_v0_pview
  const int64_t v45 = 1;
  // pto: %output__ssa_v0_pview
  const int64_t v46 = 1;
  // pto: %output__ssa_v0_pview
  int64_t v47 = v27 * v5;
  // pto: %output__ssa_v0_pview
  int64_t v48 = v46 * v47;
  // pto: %output__ssa_v0_pview
  pto::Shape<1, 1, 1, -1, 64> v49 = pto::Shape<1, 1, 1, -1, 64>(v44, v45, v46, v27, v5);
  // pto: %output__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v50 = pto::Stride<-1, -1, -1, -1, -1>(v45 * v48, v48, v47, v5, v6);
  // pto: %output__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v51 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v43 + (v42 + v8 * v5 + v8 * v6), v49, v50);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v51, v40);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}