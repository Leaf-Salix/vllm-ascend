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

AICORE void kernel(__gm__ float* v1, __gm__ float* v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v3 = 16384;
  const int64_t v4 = 0;
  const int64_t v5 = 32768;
  // pto: %c64_index
  const int64_t v6 = 64;
  // pto: %c256_index
  const int64_t v7 = 256;
  // pto: %c1_index
  const int64_t v8 = 1;
  // pto: %c0_index
  const int64_t v9 = 0;
  // pto: %c4_index
  const int64_t v10 = 4;
  // pto: %c2_index
  const int64_t v11 = 2;
  // pto: %a__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %a__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %a__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %a__ssa_v0_view
  int64_t v15 = v7 * v6;
  // pto: %a__ssa_v0_view
  int64_t v16 = v14 * v15;
  // pto: %a__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v7, v6);
  // pto: %a__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v6, v8);
  // pto: %a__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v17, v18);
  // pto: %output__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %output__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %output__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %output__ssa_v0_view
  int64_t v23 = v6 * v6;
  // pto: %output__ssa_v0_view
  int64_t v24 = v22 * v23;
  // pto: %output__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v20, v21, v22, v6, v6);
  // pto: %output__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v21 * v24, v24, v23, v6, v8);
  // pto: %output__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v25, v26);
  // pto: %seed__ssa_v0
  Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v6, v6);
  // pto: %seed__ssa_v0
  uint64_t v29 = (uint64_t) v5;
  TASSIGN(v28, v29);
  // pto: %a__ssa_v0_pview
  __gm__ float* v30 = PTOAS__GLOBAL_TENSOR_DATA(v19);
  // pto: %a__ssa_v0_pview
  pto::Shape<1, 1, 1, 64, 64> v31 = pto::Shape<1, 1, 1, 64, 64>();
  // pto: %a__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 64, 1> v32 = pto::Stride<4096, 4096, 4096, 64, 1>();
  // pto: %a__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v30, v31, v32);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  TLOAD(v28, v33);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  for (int64_t v34 = v9; v34 < v10; v34 += v8) {
    // pto: %0
    ;
    int64_t v35 = v34 % v11;
    // pto: %1, %2, %3
    ;
    int64_t v36 = v35 < v9 ? (int64_t) ((uint64_t) v35 + (uint64_t) v11) : v35;
    // pto: %t__ssa_v0
    ;
    int64_t v37 = v36 == v8 ? v3 : v4;
    // pto: %t__ssa_v0
    ;
    Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v6, v6);
    // pto: %t__ssa_v0
    ;
    uint64_t v39 = (uint64_t) v37;
    TASSIGN(v38, v39);
    // pto: %4
    ;
    int64_t v40 = (int64_t) ((uint64_t) v34 * (uint64_t) v6);
    // pto: %5
    ;
    int64_t v41 = v40 < v9 ? v9 : v40;
    // pto: %6
    ;
    __gm__ float* v42 = PTOAS__GLOBAL_TENSOR_DATA(v19);
    // pto: %6
    ;
    const int64_t v43 = 0;
    // pto: %6
    ;
    const int64_t v44 = 64;
    // pto: %6
    ;
    pto::Shape<1, 1, 1, 64, 64> v45 = pto::Shape<1, 1, 1, 64, 64>();
    // pto: %6
    ;
    pto::Stride<4096, 4096, 4096, 64, 1> v46 = pto::Stride<4096, 4096, 4096, 64, 1>();
    // pto: %6
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v42 + (v43 + v41 * v44), v45, v46);
    int64_t v48 = (int64_t) ((uint64_t) v36 % (uint64_t) v11) == v8 ? v8 : v9;
    event_t v49 = (event_t) v48;
    wait_flag(PIPE_V, PIPE_MTE2, v49);
    TLOAD(v38, v47);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TADD(v28, v28, v38);
    event_t v50 = (event_t) v48;
    set_flag(PIPE_V, PIPE_MTE2, v50);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %output__ssa_v0_pview
  __gm__ float* v51 = PTOAS__GLOBAL_TENSOR_DATA(v27);
  // pto: %output__ssa_v0_pview
  pto::Shape<1, 1, 1, 64, 64> v52 = pto::Shape<1, 1, 1, 64, 64>();
  // pto: %output__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 64, 1> v53 = pto::Stride<4096, 4096, 4096, 64, 1>();
  // pto: %output__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v51, v52, v53);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v54, v28);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}