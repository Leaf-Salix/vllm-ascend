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

AICORE void manual_safe_id(__gm__ float* v1, __gm__ float* v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %one
  const int64_t v3 = 1;
  // pto: %cols
  const int64_t v4 = 32;
  // pto: %addr
  const int64_t v5 = 0;
  // pto: %shape_src
  const int64_t v6 = 1;
  // pto: %shape_src
  const int64_t v7 = 1;
  // pto: %shape_src
  const int64_t v8 = 1;
  // pto: %shape_src
  int64_t v9 = v3 * v4;
  // pto: %shape_src
  int64_t v10 = v8 * v9;
  // pto: %shape_src
  pto::Shape<1, 1, 1, 1, 32> v11 = pto::Shape<1, 1, 1, 1, 32>(v6, v7, v8, v3, v4);
  // pto: %shape_src
  pto::Stride<-1, -1, -1, -1, -1> v12 = pto::Stride<-1, -1, -1, -1, -1>(v7 * v10, v10, v9, v4, v3);
  // pto: %shape_src
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v13 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v11, v12);
  // pto: %shape_dst
  const int64_t v14 = 1;
  // pto: %shape_dst
  const int64_t v15 = 1;
  // pto: %shape_dst
  const int64_t v16 = 1;
  // pto: %shape_dst
  int64_t v17 = v3 * v4;
  // pto: %shape_dst
  int64_t v18 = v16 * v17;
  // pto: %shape_dst
  pto::Shape<1, 1, 1, 1, 32> v19 = pto::Shape<1, 1, 1, 1, 32>(v14, v15, v16, v3, v4);
  // pto: %shape_dst
  pto::Stride<-1, -1, -1, -1, -1> v20 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v18, v18, v17, v4, v3);
  // pto: %shape_dst
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v19, v20);
  // pto: %view_src
  __gm__ float* v22 = PTOAS__GLOBAL_TENSOR_DATA(v13);
  // pto: %view_src
  pto::Shape<1, 1, 1, 1, 32> v23 = pto::Shape<1, 1, 1, 1, 32>();
  // pto: %view_src
  pto::Stride<32, 32, 32, 32, 1> v24 = pto::Stride<32, 32, 32, 32, 1>();
  // pto: %view_src
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v22, v23, v24);
  // pto: %view_dst
  __gm__ float* v26 = PTOAS__GLOBAL_TENSOR_DATA(v21);
  // pto: %view_dst
  pto::Shape<1, 1, 1, 1, 32> v27 = pto::Shape<1, 1, 1, 1, 32>();
  // pto: %view_dst
  pto::Stride<32, 32, 32, 32, 1> v28 = pto::Stride<32, 32, 32, 32, 1>();
  // pto: %view_dst
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v29 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v26, v27, v28);
  // pto: %tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v3, v4);
  // pto: %tile
  uint64_t v31 = (uint64_t) v5;
  TASSIGN(v30, v31);
  TLOAD(v30, v25);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v29, v30);
  #endif // __DAV_VEC__

  return;
}

AICORE void manual_other_pair() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  #endif // __DAV_VEC__

  return;
}

AICORE void manual_high_level() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

AICORE void manual_constant_dynamic() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %id
  const int64_t v1 = 6;
  event_t v2 = (event_t) v1;
  set_flag(PIPE_MTE2, PIPE_MTE3, v2);
  event_t v3 = (event_t) v1;
  wait_flag(PIPE_MTE2, PIPE_MTE3, v3);
  #endif // __DAV_VEC__

  return;
}

AICORE void manual_unrelated_dynamic(int64_t v1) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  event_t v2 = (event_t) v1;
  set_flag(PIPE_V, PIPE_MTE2, v2);
  event_t v3 = (event_t) v1;
  wait_flag(PIPE_V, PIPE_MTE2, v3);
  #endif // __DAV_VEC__

  return;
}

AICORE void unmarked_event7() {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
  #endif // __DAV_VEC__

  return;
}
