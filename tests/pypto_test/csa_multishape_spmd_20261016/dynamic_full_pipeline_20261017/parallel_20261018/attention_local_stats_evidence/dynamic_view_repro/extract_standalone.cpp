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

AICORE void extract_reshape_core(__gm__ float* v1, __gm__ int32_t* v2, __gm__ float* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c768_i64
  const int64_t v5 = 768;
  // pto: %c6_index
  const int64_t v6 = 6;
  // pto: %c32_index
  const int64_t v7 = 32;
  // pto: %c1_index
  const int64_t v8 = 1;
  // pto: %c0_index
  const int64_t v9 = 0;
  // pto: %src__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %src__ssa_v0_view
  int64_t v13 = v6 * v7;
  // pto: %src__ssa_v0_view
  int64_t v14 = v12 * v13;
  // pto: %src__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v10, v11, v12, v6, v7);
  // pto: %src__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v14, v14, v13, v7, v8);
  // pto: %src__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v17 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v15, v16);
  // pto: %out__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %out__ssa_v0_view
  int64_t v21 = v7 * v8;
  // pto: %out__ssa_v0_view
  int64_t v22 = v20 * v21;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v23 = pto::Shape<1, 1, 1, -1, -1>(v18, v19, v20, v7, v8);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v24 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v22, v22, v21, v8, v7);
  // pto: %out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v3, v23, v24);
  // pto: %pool__ssa_v0
  Tile<TileType::Vec, float, 6, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Vec, float, 6, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v6, v7);
  // pto: %pool__ssa_v0
  uint64_t v27 = (uint64_t) v4;
  TASSIGN(v26, v27);
  // pto: %src__ssa_v0_pview
  __gm__ float* v28 = PTOAS__GLOBAL_TENSOR_DATA(v17);
  // pto: %src__ssa_v0_pview
  pto::Shape<1, 1, 1, 6, 32> v29 = pto::Shape<1, 1, 1, 6, 32>();
  // pto: %src__ssa_v0_pview
  pto::Stride<192, 192, 192, 32, 1> v30 = pto::Stride<192, 192, 192, 32, 1>();
  // pto: %src__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 6, 32>, pto::Stride<192, 192, 192, 32, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 6, 32>, pto::Stride<192, 192, 192, 32, 1>, pto::Layout::ND>(v28, v29, v30);
  TLOAD(v26, v31);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %t__tile
  int32_t v32 = (v2)[v9];
  // pto: %selected__ssa_v0
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v7);
  // pto: %selected__ssa_v0
  uint64_t v34 = (uint64_t) v5;
  TASSIGN(v33, v34);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %0
  uint16_t v35 = static_cast<uint16_t>((int64_t) v32);
  uint16_t v36 = static_cast<uint16_t>(v9);
  TEXTRACT(v33, v26, v35, v36);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %column__ssa_v0
  Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v8);
  // pto: %column__ssa_v0
  uint64_t v38 = (uint64_t) v5;
  TASSIGN(v37, v38);
  // pto: %out__ssa_v0_pview
  __gm__ float* v39 = PTOAS__GLOBAL_TENSOR_DATA(v25);
  // pto: %out__ssa_v0_pview
  pto::Shape<1, 1, 1, 32, 1> v40 = pto::Shape<1, 1, 1, 32, 1>();
  // pto: %out__ssa_v0_pview
  pto::Stride<32, 32, 32, 1, 32> v41 = pto::Stride<32, 32, 32, 1, 32>();
  // pto: %out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<32, 32, 32, 1, 32>, pto::Layout::DN> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<32, 32, 32, 1, 32>, pto::Layout::DN>(v39, v40, v41);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v42, v37);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}