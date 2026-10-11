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
  kAsyncAivEvent7 = 2,
  kAsyncAicEvent7 = 3,
  kAsyncAivAllPipesEvent7 = 4,
  kAsyncAicAllPipesEvent7 = 5,
  kRuntimeBarrierAll = 6,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kRuntimeBarrierAll:
    // The dynamic caller executes the ordinary PIPE_ALL before publishing FIN
    // or before a same-core successor may touch pipeline resources.
    break;
  // Every event targets Scalar, so internal uses of the same pair/ID have
  // already been consumed before Scalar can issue these tail markers.
  // The caller must join all listed engines before reusing this core.
  case PTOAutoSyncTailMode::kAsyncAicAllPipesEvent7:
    set_flag(PIPE_M, PIPE_S, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAivAllPipesEvent7:
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAicEvent7:
    // The caller drains Cube and FIX independently. Join every upstream
    // matrix pipeline without making Scalar wait for device completion.
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    set_flag(PIPE_M, PIPE_S, EVENT_ID7);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAivEvent7:
    // Join MTE2/V/MTE3 without a Scalar wait. The caller owns the final
    // MTE3->S event and must consume it before publishing completion.
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // Diagnostic completion marker: the caller measures Vector separately
    // from MTE3 so a trailing store cannot be mistaken for compute overlap.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    break;
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

extern "C" AICORE void pypto_keep_prefix(int64_t);
extern "C" AICORE void pypto_wait_pre_task_end(__gm__ int64_t*, int64_t);
extern "C" AICORE void pypto_set_next_task_start(__gm__ int64_t*);
AICORE void late_producer(__gm__ int64_t* v1, __gm__ half* v2, __gm__ float* v3, int64_t v4, __gm__ int64_t* v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_VEC__)
  // pto: %c64_i64
  const int64_t v9 = 64;
  // pto: %c32832_i64
  const int64_t v10 = 32832;
  // pto: %c32_index
  const int64_t v11 = 32;
  // pto: %c1_index
  const int64_t v12 = 1;
  // pto: %c3088_index
  const int64_t v13 = 3088;
  // pto: %c256_index
  const int64_t v14 = 256;
  // pto: %c23_index
  const int64_t v15 = 23;
  // pto: %c64_index
  const int64_t v16 = 64;
  // pto: %c0_index
  const int64_t v17 = 0;
  // pto: %cst_10
  const float v18 = 1.0f;
  // pto: %x__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %x__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %x__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %x__ssa_v0_view
  int64_t v22 = v13 * v14;
  // pto: %x__ssa_v0_view
  int64_t v23 = v21 * v22;
  // pto: %x__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v13, v14);
  // pto: %x__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v23, v23, v22, v14, v12);
  // pto: %x__ssa_v0_view
  GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v24, v25);
  // pto: %a_out__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %a_out__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %a_out__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %a_out__ssa_v0_view
  int64_t v30 = v13 * v14;
  // pto: %a_out__ssa_v0_view
  int64_t v31 = v29 * v30;
  // pto: %a_out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v27, v28, v29, v13, v14);
  // pto: %a_out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v28 * v31, v31, v30, v14, v12);
  // pto: %a_out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v32, v33);
  // pto: %block__ssa_v0
  int64_t v35 = (int64_t) v7;
  pypto_keep_prefix(v35);
  int64_t v36 = pypto_wait_local_boundary(v5, v6);
  const int64_t v37 = 0;
  if (v36 == v37) {
    return;
  }
  // pto: %flat_offset_mul, %flat_offset
  int64_t v38 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v35 * (uint64_t) v11) + (uint64_t) v15);
  (v1)[v38] = v35;
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb(DSB_DDR);
  // pto: %t__tmp_v0
  Tile<TileType::Vec, half, 64, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Vec, half, 64, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %t__tmp_v0
  uint64_t v40 = (uint64_t) v9;
  TASSIGN(v39, v40);
  // pto: %1
  int64_t v41 = (int64_t) ((uint64_t) v35 * (uint64_t) v16);
  // pto: %2
  int64_t v42 = v41 < v17 ? v17 : v41;
  // pto: %x__ssa_v0_pview
  __gm__ half* v43 = PTOAS__GLOBAL_TENSOR_DATA(v26);
  // pto: %x__ssa_v0_pview
  const int64_t v44 = 0;
  // pto: %x__ssa_v0_pview
  const int64_t v45 = 256;
  // pto: %x__ssa_v0_pview
  pto::Shape<1, 1, 1, 64, 256> v46 = pto::Shape<1, 1, 1, 64, 256>();
  // pto: %x__ssa_v0_pview
  pto::Stride<16384, 16384, 16384, 256, 1> v47 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %x__ssa_v0_pview
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v43 + (v44 + v42 * v45), v46, v47);
  TLOAD(v39, v48);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %value__ssa_v0
  Tile<TileType::Vec, float, 64, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 64, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
  // pto: %value__ssa_v0
  uint64_t v50 = (uint64_t) v10;
  TASSIGN(v49, v50);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  RoundMode v51 = RoundMode::CAST_ROUND;
  SaturationMode v52 = SaturationMode::ON;
  TCVT(v49, v39, v51, v52);
  for (int64_t v53 = v17; v53 < v4; v53 += v12) {
    // pto: %value__ssa_v3
    ;
    Tile<TileType::Vec, float, 64, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, float, 64, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %value__ssa_v3
    ;
    uint64_t v55 = (uint64_t) v10;
    TASSIGN(v54, v55);
    pipe_barrier(PIPE_V);
    TADDS(v54, v49, v18);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %a_out__ssa_v0_pview
  __gm__ float* v56 = PTOAS__GLOBAL_TENSOR_DATA(v34);
  // pto: %a_out__ssa_v0_pview
  const int64_t v57 = 0;
  // pto: %a_out__ssa_v0_pview
  const int64_t v58 = 256;
  // pto: %a_out__ssa_v0_pview
  pto::Shape<1, 1, 1, 64, 256> v59 = pto::Shape<1, 1, 1, 64, 256>();
  // pto: %a_out__ssa_v0_pview
  pto::Stride<16384, 16384, 16384, 256, 1> v60 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %a_out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v56 + (v57 + v42 * v58), v59, v60);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v61, v49);
  pypto_set_next_task_start(v5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}
