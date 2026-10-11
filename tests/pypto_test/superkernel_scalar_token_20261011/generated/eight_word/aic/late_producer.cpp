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
AICORE void late_producer(__gm__ int64_t* v1, __gm__ half* v2, __gm__ half* v3, __gm__ float* v4, int64_t v5, __gm__ int64_t* v6, int64_t v7, int32_t v8, int32_t v9) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c64_i64
  const int64_t v10 = 64;
  // pto: %c32832_i64
  const int64_t v11 = 32832;
  // pto: %c0_i64
  const int64_t v12 = 0;
  // pto: %c32_index
  const int64_t v13 = 32;
  // pto: %c1_index
  const int64_t v14 = 1;
  // pto: %c3088_index
  const int64_t v15 = 3088;
  // pto: %c256_index
  const int64_t v16 = 256;
  // pto: %c128_index
  const int64_t v17 = 128;
  // pto: %c23_index
  const int64_t v18 = 23;
  // pto: %c0_index
  const int64_t v19 = 0;
  // pto: %x__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %x__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %x__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %x__ssa_v0_view
  int64_t v23 = v15 * v16;
  // pto: %x__ssa_v0_view
  int64_t v24 = v22 * v23;
  // pto: %x__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v20, v21, v22, v15, v16);
  // pto: %x__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v21 * v24, v24, v23, v16, v14);
  // pto: %x__ssa_v0_view
  GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v25, v26);
  // pto: %w__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %w__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %w__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %w__ssa_v0_view
  int64_t v31 = v17 * v16;
  // pto: %w__ssa_v0_view
  int64_t v32 = v30 * v31;
  // pto: %w__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v33 = pto::Shape<1, 1, 1, -1, -1>(v28, v29, v30, v17, v16);
  // pto: %w__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v34 = pto::Stride<-1, -1, -1, -1, -1>(v29 * v32, v32, v31, v16, v14);
  // pto: %w__ssa_v0_view
  GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v33, v34);
  // pto: %a_out__ssa_v0_view
  const int64_t v36 = 1;
  // pto: %a_out__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %a_out__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %a_out__ssa_v0_view
  int64_t v39 = v15 * v16;
  // pto: %a_out__ssa_v0_view
  int64_t v40 = v38 * v39;
  // pto: %a_out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v15, v16);
  // pto: %a_out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v16, v14);
  // pto: %a_out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v41, v42);
  // pto: %block__ssa_v0
  int64_t v44 = (int64_t) v8;
  pypto_keep_prefix(v44);
  int64_t v45 = pypto_wait_local_boundary(v6, v7);
  const int64_t v46 = 0;
  if (v45 == v46) {
    return;
  }
  // pto: %flat_offset_mul, %flat_offset
  int64_t v47 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v44 * (uint64_t) v13) + (uint64_t) v18);
  (v1)[v47] = v44;
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb(DSB_DDR);
  // pto: %a_l1__ssa_v0
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v48 = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v17, v17);
  // pto: %a_l1__ssa_v0
  uint64_t v49 = (uint64_t) v10;
  TASSIGN(v48, v49);
  // pto: %1
  int64_t v50 = (int64_t) ((uint64_t) v44 * (uint64_t) v17);
  // pto: %2
  int64_t v51 = v50 < v19 ? v19 : v50;
  // pto: %x__ssa_v0_pview
  __gm__ half* v52 = PTOAS__GLOBAL_TENSOR_DATA(v27);
  // pto: %x__ssa_v0_pview
  const int64_t v53 = 0;
  // pto: %x__ssa_v0_pview
  const int64_t v54 = 256;
  // pto: %x__ssa_v0_pview
  pto::Shape<1, 1, 1, 128, 128> v55 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %x__ssa_v0_pview
  pto::Stride<32768, 32768, 32768, 256, 1> v56 = pto::Stride<32768, 32768, 32768, 256, 1>();
  // pto: %x__ssa_v0_pview
  GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND> v57 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND>(v52 + (v53 + v51 * v54), v55, v56);
  TLOAD(v48, v57);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  // pto: %b_l1__ssa_v0
  Tile<TileType::Mat, half, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Mat, half, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v17, v16);
  // pto: %b_l1__ssa_v0
  uint64_t v59 = (uint64_t) v11;
  TASSIGN(v58, v59);
  // pto: %w__ssa_v0_pview
  __gm__ half* v60 = PTOAS__GLOBAL_TENSOR_DATA(v35);
  // pto: %w__ssa_v0_pview
  pto::Shape<1, 1, 1, 128, 256> v61 = pto::Shape<1, 1, 1, 128, 256>();
  // pto: %w__ssa_v0_pview
  pto::Stride<32768, 32768, 32768, 256, 1> v62 = pto::Stride<32768, 32768, 32768, 256, 1>();
  // pto: %w__ssa_v0_pview
  GlobalTensor<half, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND> v63 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND>(v60, v61, v62);
  TLOAD(v58, v63);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  // pto: %a__ssa_v0
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v17, v17);
  // pto: %a__ssa_v0
  uint64_t v65 = (uint64_t) v12;
  TASSIGN(v64, v65);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v64, v48);
  // pto: %b__ssa_v0
  Tile<TileType::Right, half, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Right, half, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v17, v16);
  // pto: %b__ssa_v0
  uint64_t v67 = (uint64_t) v12;
  TASSIGN(v66, v67);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v66, v58);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  // pto: %acc__ssa_v0
  Tile<TileType::Acc, float, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Acc, float, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v17, v16);
  // pto: %acc__ssa_v0
  uint64_t v69 = (uint64_t) v12;
  TASSIGN(v68, v69);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v68, v64, v66);
  for (int64_t v70 = v19; v70 < v5; v70 += v14) {
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v68, v68, v64, v66);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %a_out__ssa_v0_pview
  __gm__ float* v71 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %a_out__ssa_v0_pview
  const int64_t v72 = 0;
  // pto: %a_out__ssa_v0_pview
  const int64_t v73 = 256;
  // pto: %a_out__ssa_v0_pview
  pto::Shape<1, 1, 1, 128, 256> v74 = pto::Shape<1, 1, 1, 128, 256>();
  // pto: %a_out__ssa_v0_pview
  pto::Stride<32768, 32768, 32768, 256, 1> v75 = pto::Stride<32768, 32768, 32768, 256, 1>();
  // pto: %a_out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND>(v71 + (v72 + v51 * v73), v74, v75);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v76, v68);
  pypto_set_next_task_start(v6);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}
