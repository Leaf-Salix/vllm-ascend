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

AICORE void runtime_scalar_token_vector(__gm__ float* v1, __gm__ float* v2) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const float v3 = 1.0f;
  // pto: %zero
  const int64_t v4 = 0;
  // pto: %one
  const int64_t v5 = 1;
  // pto: %cols
  const int64_t v6 = 32;
  // pto: %addr
  const int64_t v7 = 0;
  // pto: %shape_src
  const int64_t v8 = 1;
  // pto: %shape_src
  const int64_t v9 = 1;
  // pto: %shape_src
  const int64_t v10 = 1;
  // pto: %shape_src
  int64_t v11 = v5 * v6;
  // pto: %shape_src
  int64_t v12 = v10 * v11;
  // pto: %shape_src
  pto::Shape<1, 1, 1, 1, 32> v13 = pto::Shape<1, 1, 1, 1, 32>(v8, v9, v10, v5, v6);
  // pto: %shape_src
  pto::Stride<-1, -1, -1, -1, -1> v14 = pto::Stride<-1, -1, -1, -1, -1>(v9 * v12, v12, v11, v6, v5);
  // pto: %shape_src
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v15 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v13, v14);
  // pto: %shape_dst
  const int64_t v16 = 1;
  // pto: %shape_dst
  const int64_t v17 = 1;
  // pto: %shape_dst
  const int64_t v18 = 1;
  // pto: %shape_dst
  int64_t v19 = v5 * v6;
  // pto: %shape_dst
  int64_t v20 = v18 * v19;
  // pto: %shape_dst
  pto::Shape<1, 1, 1, 1, 32> v21 = pto::Shape<1, 1, 1, 1, 32>(v16, v17, v18, v5, v6);
  // pto: %shape_dst
  pto::Stride<-1, -1, -1, -1, -1> v22 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v20, v20, v19, v6, v5);
  // pto: %shape_dst
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v21, v22);
  // pto: %view_src
  __gm__ float* v24 = PTOAS__GLOBAL_TENSOR_DATA(v15);
  // pto: %view_src
  pto::Shape<1, 1, 1, 1, 32> v25 = pto::Shape<1, 1, 1, 1, 32>();
  // pto: %view_src
  pto::Stride<32, 32, 32, 32, 1> v26 = pto::Stride<32, 32, 32, 32, 1>();
  // pto: %view_src
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v27 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v24, v25, v26);
  // pto: %view_dst
  __gm__ float* v28 = PTOAS__GLOBAL_TENSOR_DATA(v23);
  // pto: %view_dst
  pto::Shape<1, 1, 1, 1, 32> v29 = pto::Shape<1, 1, 1, 1, 32>();
  // pto: %view_dst
  pto::Stride<32, 32, 32, 32, 1> v30 = pto::Stride<32, 32, 32, 32, 1>();
  // pto: %view_dst
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v28, v29, v30);
  // pto: %tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5, v6);
  // pto: %tile
  uint64_t v33 = (uint64_t) v7;
  TASSIGN(v32, v33);
  TLOAD(v32, v27);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  v32.SetValue(v4, v3);
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  TSTORE(v31, v32);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}
