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

// PYPTO_TASK_BOUNDARY_JOINS_V1 loop_join mask=7
#ifndef PYPTO_TASK_BOUNDARY_JOINS_V1
#error reduced tail joins require a matching PyPTO wrapper
#endif
AICORE void loop_join(__gm__ half* v1, __gm__ float* v2, int64_t v3, bool v4) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %zero
  const int64_t v5 = 0;
  // pto: %one
  const int64_t v6 = 1;
  // pto: %n
  const int64_t v7 = 16;
  // pto: %addr
  const int64_t v8 = 512;
  // pto: %sv
  const int64_t v9 = 1;
  // pto: %sv
  const int64_t v10 = 1;
  // pto: %sv
  const int64_t v11 = 1;
  // pto: %sv
  int64_t v12 = v7 * v7;
  // pto: %sv
  int64_t v13 = v11 * v12;
  // pto: %sv
  pto::Shape<1, 1, 1, 16, 16> v14 = pto::Shape<1, 1, 1, 16, 16>(v9, v10, v11, v7, v7);
  // pto: %sv
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v7, v6);
  // pto: %sv
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %dv
  const int64_t v17 = 1;
  // pto: %dv
  const int64_t v18 = 1;
  // pto: %dv
  const int64_t v19 = 1;
  // pto: %dv
  int64_t v20 = v7 * v7;
  // pto: %dv
  int64_t v21 = v19 * v20;
  // pto: %dv
  pto::Shape<1, 1, 1, 16, 16> v22 = pto::Shape<1, 1, 1, 16, 16>(v17, v18, v19, v7, v7);
  // pto: %dv
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v7, v6);
  // pto: %dv
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v22, v23);
  // pto: %s
  __gm__ half* v25 = PTOAS__GLOBAL_TENSOR_DATA(v16);
  // pto: %s
  pto::Shape<1, 1, 1, 16, 16> v26 = pto::Shape<1, 1, 1, 16, 16>();
  // pto: %s
  pto::Stride<256, 256, 256, 16, 1> v27 = pto::Stride<256, 256, 256, 16, 1>();
  // pto: %s
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v25, v26, v27);
  // pto: %d
  __gm__ float* v29 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %d
  pto::Shape<1, 1, 1, 16, 16> v30 = pto::Shape<1, 1, 1, 16, 16>();
  // pto: %d
  pto::Stride<256, 256, 256, 16, 1> v31 = pto::Stride<256, 256, 256, 16, 1>();
  // pto: %d
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v29, v30, v31);
  // pto: %l1
  Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v33;
  // pto: %l1
  uint64_t v34 = (uint64_t) v8;
  TASSIGN(v33, v34);
  // pto: %a
  Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v35;
  // pto: %a
  uint64_t v36 = (uint64_t) v8;
  TASSIGN(v35, v36);
  // pto: %b
  Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v37;
  // pto: %b
  uint64_t v38 = (uint64_t) v8;
  TASSIGN(v37, v38);
  // pto: %c
  Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v39;
  // pto: %c
  uint64_t v40 = (uint64_t) v8;
  TASSIGN(v39, v40);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int64_t v41 = v5; v41 < v3; v41 += v6) {
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v33, v28);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v35, v33);
    TMOV(v37, v33);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v39, v35, v37);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v32, v39);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}

AICORE void later_fix_work(__gm__ half* v1, __gm__ float* v2, int64_t v3, bool v4) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %zero
  const int64_t v5 = 0;
  // pto: %one
  const int64_t v6 = 1;
  // pto: %n
  const int64_t v7 = 16;
  // pto: %addr
  const int64_t v8 = 512;
  // pto: %sv
  const int64_t v9 = 1;
  // pto: %sv
  const int64_t v10 = 1;
  // pto: %sv
  const int64_t v11 = 1;
  // pto: %sv
  int64_t v12 = v7 * v7;
  // pto: %sv
  int64_t v13 = v11 * v12;
  // pto: %sv
  pto::Shape<1, 1, 1, 16, 16> v14 = pto::Shape<1, 1, 1, 16, 16>(v9, v10, v11, v7, v7);
  // pto: %sv
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v7, v6);
  // pto: %sv
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %dv
  const int64_t v17 = 1;
  // pto: %dv
  const int64_t v18 = 1;
  // pto: %dv
  const int64_t v19 = 1;
  // pto: %dv
  int64_t v20 = v7 * v7;
  // pto: %dv
  int64_t v21 = v19 * v20;
  // pto: %dv
  pto::Shape<1, 1, 1, 16, 16> v22 = pto::Shape<1, 1, 1, 16, 16>(v17, v18, v19, v7, v7);
  // pto: %dv
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v7, v6);
  // pto: %dv
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v22, v23);
  // pto: %s
  __gm__ half* v25 = PTOAS__GLOBAL_TENSOR_DATA(v16);
  // pto: %s
  pto::Shape<1, 1, 1, 16, 16> v26 = pto::Shape<1, 1, 1, 16, 16>();
  // pto: %s
  pto::Stride<256, 256, 256, 16, 1> v27 = pto::Stride<256, 256, 256, 16, 1>();
  // pto: %s
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v25, v26, v27);
  // pto: %d
  __gm__ float* v29 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %d
  pto::Shape<1, 1, 1, 16, 16> v30 = pto::Shape<1, 1, 1, 16, 16>();
  // pto: %d
  pto::Stride<256, 256, 256, 16, 1> v31 = pto::Stride<256, 256, 256, 16, 1>();
  // pto: %d
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v29, v30, v31);
  // pto: %l1
  Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v33;
  // pto: %l1
  uint64_t v34 = (uint64_t) v8;
  TASSIGN(v33, v34);
  // pto: %a
  Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v35;
  // pto: %a
  uint64_t v36 = (uint64_t) v8;
  TASSIGN(v35, v36);
  // pto: %b
  Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v37;
  // pto: %b
  uint64_t v38 = (uint64_t) v8;
  TASSIGN(v37, v38);
  // pto: %c
  Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v39;
  // pto: %c
  uint64_t v40 = (uint64_t) v8;
  TASSIGN(v39, v40);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int64_t v41 = v5; v41 < v3; v41 += v6) {
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v33, v28);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v35, v33);
    TMOV(v37, v33);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v39, v35, v37);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v32, v39);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  if (v4) {
    pipe_barrier(PIPE_FIX);
    TSTORE(v32, v39);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}
