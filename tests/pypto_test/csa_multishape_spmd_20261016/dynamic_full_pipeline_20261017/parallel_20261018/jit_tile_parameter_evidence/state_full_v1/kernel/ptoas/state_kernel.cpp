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

AICORE void state_kernel(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int64_t v4) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v5 = 0;
  // pto: %c12288_i64
  const int64_t v6 = 12288;
  // pto: %c36864_i64
  const int64_t v7 = 36864;
  // pto: %c1024_index
  const int64_t v8 = 1024;
  // pto: %c1_index
  const int64_t v9 = 1;
  // pto: %c3_index
  const int64_t v10 = 3;
  // pto: %cst_6
  const float v11 = 0.0f;
  // pto: %c6_index
  const int64_t v12 = 6;
  // pto: %c0_index
  const int64_t v13 = 0;
  // pto: %c2_index
  const int64_t v14 = 2;
  // pto: %source__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %source__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %source__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %source__ssa_v0_view
  int64_t v18 = (int64_t) v4;
  // pto: %source__ssa_v0_view
  int64_t v19 = v18 * v8;
  // pto: %source__ssa_v0_view
  int64_t v20 = v17 * v19;
  // pto: %source__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v21 = pto::Shape<1, 1, 1, -1, -1>(v15, v16, v17, v18, v8);
  // pto: %source__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v22 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v20, v20, v19, v8, v9);
  // pto: %source__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v21, v22);
  // pto: %roots_out__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %roots_out__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %roots_out__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %roots_out__ssa_v0_view
  int64_t v27 = (int64_t) v4;
  // pto: %roots_out__ssa_v0_view
  int64_t v28 = v27 * v8;
  // pto: %roots_out__ssa_v0_view
  int64_t v29 = v26 * v28;
  // pto: %roots_out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v27, v8);
  // pto: %roots_out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v29, v29, v28, v8, v9);
  // pto: %roots_out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v30, v31);
  // pto: %segments_out__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %segments_out__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %segments_out__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %segments_out__ssa_v0_view
  int64_t v36 = (int64_t) v4;
  // pto: %segments_out__ssa_v0_view
  int64_t v37 = v36 * v8;
  // pto: %segments_out__ssa_v0_view
  int64_t v38 = v35 * v37;
  // pto: %segments_out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v39 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v36, v8);
  // pto: %segments_out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v40 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v38, v38, v37, v8, v9);
  // pto: %segments_out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v39, v40);
  // pto: %roots__ssa_v0
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v8);
  // pto: %roots__ssa_v0
  uint64_t v43 = (uint64_t) v5;
  TASSIGN(v42, v43);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TEXPANDS(v42, v11);
  // pto: %segments__ssa_v0
  Tile<TileType::Vec, float, 6, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 6, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v8);
  // pto: %segments__ssa_v0
  uint64_t v45 = (uint64_t) v6;
  TASSIGN(v44, v45);
  TEXPANDS(v44, v11);
  // pto: %0
  for (int64_t v46 = v13; v46 < (v4 < v14 ? v4 : v14); v46 += v9) {
    // pto: %incoming__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v8);
    // pto: %incoming__ssa_v0
    ;
    uint64_t v48 = (uint64_t) v7;
    TASSIGN(v47, v48);
    // pto: %1
    ;
    int64_t v49 = v46 < v13 ? v13 : v46;
    // pto: %source__ssa_v0_pview
    ;
    __gm__ float* v50 = PTOAS__GLOBAL_TENSOR_DATA(v23);
    // pto: %source__ssa_v0_pview
    ;
    const int64_t v51 = 0;
    // pto: %source__ssa_v0_pview
    ;
    const int64_t v52 = 1024;
    // pto: %source__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v53 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %source__ssa_v0_pview
    ;
    pto::Stride<1024, 1024, 1024, 1024, 1> v54 = pto::Stride<1024, 1024, 1024, 1024, 1>();
    // pto: %source__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v50 + (v51 + v49 * v52), v53, v54);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v47, v55);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %assemble_view
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v56;
    // pto: %assemble_view
    ;
    uint64_t v57 = (uint64_t) v5;
    TASSIGN(v56, v57);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TMOV(v56, v47);
    // pto: %2
    ;
    Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58;
    // pto: %2
    ;
    uint64_t v59 = (uint64_t) v6;
    TASSIGN(v58, v59);
    TMOV(v58, v47);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %next_roots__ssa_v0
    ;
    Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v60 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v8);
    // pto: %next_roots__ssa_v0
    ;
    uint64_t v61 = (uint64_t) v5;
    TASSIGN(v60, v61);
    // pto: %next_segments__ssa_v0
    ;
    Tile<TileType::Vec, float, 6, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, float, 6, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v8);
    // pto: %next_segments__ssa_v0
    ;
    uint64_t v63 = (uint64_t) v6;
    TASSIGN(v62, v63);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %3
  int64_t v64 = v4 < v10 ? v4 : v10;
  v42.SetValidShape(v64, v8);
  // pto: %roots_out__ssa_v0_pview
  const int64_t v65 = 0;
  // pto: %roots_out__ssa_v0_pview
  __gm__ float* v66 = PTOAS__GLOBAL_TENSOR_DATA(v32);
  // pto: %roots_out__ssa_v0_pview
  const int64_t v67 = 1;
  // pto: %roots_out__ssa_v0_pview
  const int64_t v68 = 1;
  // pto: %roots_out__ssa_v0_pview
  const int64_t v69 = 1;
  // pto: %roots_out__ssa_v0_pview
  int64_t v70 = v64 * v8;
  // pto: %roots_out__ssa_v0_pview
  int64_t v71 = v69 * v70;
  // pto: %roots_out__ssa_v0_pview
  pto::Shape<1, 1, 1, -1, 1024> v72 = pto::Shape<1, 1, 1, -1, 1024>(v67, v68, v69, v64, v8);
  // pto: %roots_out__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v73 = pto::Stride<-1, -1, -1, -1, -1>(v68 * v71, v71, v70, v8, v9);
  // pto: %roots_out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v66 + (v65 + v13 * v8 + v13 * v9), v72, v73);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v74, v42);
  // pto: %5
  int64_t v75 = v4 < v12 ? v4 : v12;
  v44.SetValidShape(v75, v8);
  // pto: %segments_out__ssa_v0_pview
  const int64_t v76 = 0;
  // pto: %segments_out__ssa_v0_pview
  __gm__ float* v77 = PTOAS__GLOBAL_TENSOR_DATA(v41);
  // pto: %segments_out__ssa_v0_pview
  const int64_t v78 = 1;
  // pto: %segments_out__ssa_v0_pview
  const int64_t v79 = 1;
  // pto: %segments_out__ssa_v0_pview
  const int64_t v80 = 1;
  // pto: %segments_out__ssa_v0_pview
  int64_t v81 = v75 * v8;
  // pto: %segments_out__ssa_v0_pview
  int64_t v82 = v80 * v81;
  // pto: %segments_out__ssa_v0_pview
  pto::Shape<1, 1, 1, -1, 1024> v83 = pto::Shape<1, 1, 1, -1, 1024>(v78, v79, v80, v75, v8);
  // pto: %segments_out__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v84 = pto::Stride<-1, -1, -1, -1, -1>(v79 * v82, v82, v81, v8, v9);
  // pto: %segments_out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v85 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 1024>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v77 + (v76 + v13 * v8 + v13 * v9), v83, v84);
  TSTORE(v85, v44);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}