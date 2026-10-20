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

AICORE void compare_kernel(__gm__ float* v1, __gm__ int32_t* v2, __gm__ int32_t* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c2048_i64
  const int64_t v5 = 2048;
  // pto: %c6144_i64
  const int64_t v6 = 6144;
  // pto: %c8192_i64
  const int64_t v7 = 8192;
  // pto: %c8_index
  const int64_t v8 = 8;
  // pto: %c64_index
  const int64_t v9 = 64;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c4096_index
  const int64_t v11 = 4096;
  // pto: %cst_9
  const float v12 = 0.5f;
  // pto: %src__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %src__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %src__ssa_v0_view
  int64_t v16 = v8 * v9;
  // pto: %src__ssa_v0_view
  int64_t v17 = v15 * v16;
  // pto: %src__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v13, v14, v15, v8, v9);
  // pto: %src__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v14 * v17, v17, v16, v9, v10);
  // pto: %src__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v20 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v18, v19);
  // pto: %selected__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %selected__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %selected__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %selected__ssa_v0_view
  int64_t v24 = v8 * v9;
  // pto: %selected__ssa_v0_view
  int64_t v25 = v23 * v24;
  // pto: %selected__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v26 = pto::Shape<1, 1, 1, -1, -1>(v21, v22, v23, v8, v9);
  // pto: %selected__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v27 = pto::Stride<-1, -1, -1, -1, -1>(v22 * v25, v25, v24, v9, v10);
  // pto: %selected__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v28 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v26, v27);
  // pto: %counts__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %counts__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %counts__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %counts__ssa_v0_view
  int64_t v32 = v10 * v8;
  // pto: %counts__ssa_v0_view
  int64_t v33 = v31 * v32;
  // pto: %counts__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v34 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v10, v8);
  // pto: %counts__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v35 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v33, v33, v32, v8, v10);
  // pto: %counts__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v36 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v34, v35);
  // pto: %values__ssa_v0
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v9);
  // pto: %values__ssa_v0
  uint64_t v38 = (uint64_t) v4;
  TASSIGN(v37, v38);
  // pto: %src__ssa_v0_pview
  __gm__ float* v39 = PTOAS__GLOBAL_TENSOR_DATA(v20);
  // pto: %src__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 64> v40 = pto::Shape<1, 1, 1, 8, 64>();
  // pto: %src__ssa_v0_pview
  pto::Stride<512, 512, 512, 64, 1> v41 = pto::Stride<512, 512, 512, 64, 1>();
  // pto: %src__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v39, v40, v41);
  TLOAD(v37, v42);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %tmp__ssa_v0
  Tile<TileType::Vec, uint8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, uint8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
  // pto: %tmp__ssa_v0
  uint64_t v44 = (uint64_t) v5;
  TASSIGN(v43, v44);
  // pto: %ids__ssa_v0
  Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v9);
  // pto: %ids__ssa_v0
  uint64_t v46 = (uint64_t) v6;
  TASSIGN(v45, v46);
  // pto: %count__ssa_v0
  Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v8);
  // pto: %count__ssa_v0
  uint64_t v48 = (uint64_t) v7;
  TASSIGN(v47, v48);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  const int32_t v49 = 0;
  TGATHER<Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, uint8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, CmpMode::GT>(v45, v37, v12, v43, v47, v49);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  // pto: %selected__ssa_v0_pview
  __gm__ int32_t* v50 = PTOAS__GLOBAL_TENSOR_DATA(v28);
  // pto: %selected__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 64> v51 = pto::Shape<1, 1, 1, 8, 64>();
  // pto: %selected__ssa_v0_pview
  pto::Stride<512, 512, 512, 64, 1> v52 = pto::Stride<512, 512, 512, 64, 1>();
  // pto: %selected__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v53 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v50, v51, v52);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v53, v45);
  // pto: %counts__ssa_v0_pview
  __gm__ int32_t* v54 = PTOAS__GLOBAL_TENSOR_DATA(v36);
  // pto: %counts__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 8> v55 = pto::Shape<1, 1, 1, 1, 8>();
  // pto: %counts__ssa_v0_pview
  pto::Stride<8, 8, 8, 8, 1> v56 = pto::Stride<8, 8, 8, 8, 1>();
  // pto: %counts__ssa_v0_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v57 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v54, v55, v56);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  TSTORE(v57, v47);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}