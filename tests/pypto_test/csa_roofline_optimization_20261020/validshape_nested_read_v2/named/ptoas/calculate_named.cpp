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

AICORE void calculate_named(__gm__ float* v1, __gm__ int32_t* v2, __gm__ float* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c1_index
  const int64_t v5 = 1;
  // pto: %c2048_index
  const int64_t v6 = 2048;
  // pto: %c0_index
  const int64_t v7 = 0;
  // pto: %cst_4
  const float v8 = 1.0f;
  // pto: %src__ssa_v0_view
  const int64_t v9 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %src__ssa_v0_view
  int64_t v12 = v5 * v6;
  // pto: %src__ssa_v0_view
  int64_t v13 = v11 * v12;
  // pto: %src__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v14 = pto::Shape<1, 1, 1, -1, -1>(v9, v10, v11, v5, v6);
  // pto: %src__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v6, v5);
  // pto: %src__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %out__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %out__ssa_v0_view
  int64_t v20 = v5 * v6;
  // pto: %out__ssa_v0_view
  int64_t v21 = v19 * v20;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v17, v18, v19, v5, v6);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v6, v5);
  // pto: %out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v22, v23);
  // pto: %tile__ssa_v0
  Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5, v6);
  // pto: %tile__ssa_v0
  uint64_t v26 = (uint64_t) v4;
  TASSIGN(v25, v26);
  // pto: %src__ssa_v0_pview
  __gm__ float* v27 = PTOAS__GLOBAL_TENSOR_DATA(v16);
  // pto: %src__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 2048> v28 = pto::Shape<1, 1, 1, 1, 2048>();
  // pto: %src__ssa_v0_pview
  pto::Stride<2048, 2048, 2048, 2048, 1> v29 = pto::Stride<2048, 2048, 2048, 2048, 1>();
  // pto: %src__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 2048>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(v27, v28, v29);
  TLOAD(v25, v30);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %n__tile
  int32_t v31 = (v2)[v7];
  // pto: %0
  int64_t v32 = (int64_t) v31;
  v25.SetValidShape(v5, v32);
  // pto: %result__ssa_v0
  Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5, v32);
  // pto: %result__ssa_v0
  uint64_t v34 = (uint64_t) v4;
  TASSIGN(v33, v34);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADDS(v33, v25, v8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %out__ssa_v0_pview
  const int64_t v35 = 0;
  // pto: %out__ssa_v0_pview
  __gm__ float* v36 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %out__ssa_v0_pview
  const int64_t v37 = 1;
  // pto: %out__ssa_v0_pview
  const int64_t v38 = 1;
  // pto: %out__ssa_v0_pview
  const int64_t v39 = 1;
  // pto: %out__ssa_v0_pview
  int64_t v40 = v5 * v6;
  // pto: %out__ssa_v0_pview
  int64_t v41 = v39 * v40;
  // pto: %out__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, -1> v42 = pto::Shape<1, 1, 1, 1, -1>(v37, v38, v39, v5, v32);
  // pto: %out__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v6, v5);
  // pto: %out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v36 + (v35 + v7 * v6 + v7 * v5), v42, v43);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v44, v33);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}