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

AICORE void pool_incore(__gm__ half* v1, __gm__ half* v2, __gm__ float* v3) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v4 = 0;
  const int64_t v5 = 4096;
  const int64_t v6 = 8192;
  const int64_t v7 = 12288;
  const int64_t v8 = 16384;
  const int64_t v9 = 20480;
  const int64_t v10 = 24576;
  const int64_t v11 = 28672;
  // pto: %c512_index
  const int64_t v12 = 512;
  // pto: %c384_index
  const int64_t v13 = 384;
  // pto: %c1_index
  const int64_t v14 = 1;
  // pto: %c16_index
  const int64_t v15 = 16;
  // pto: %c64_index
  const int64_t v16 = 64;
  // pto: %x__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %x__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %x__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %x__ssa_v0_view
  int64_t v20 = v12 * v13;
  // pto: %x__ssa_v0_view
  int64_t v21 = v19 * v20;
  // pto: %x__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v17, v18, v19, v12, v13);
  // pto: %x__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v13, v14);
  // pto: %x__ssa_v0_view
  GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v22, v23);
  // pto: %w__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %w__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %w__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %w__ssa_v0_view
  int64_t v28 = v13 * v15;
  // pto: %w__ssa_v0_view
  int64_t v29 = v27 * v28;
  // pto: %w__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v13, v15);
  // pto: %w__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v15, v14);
  // pto: %w__ssa_v0_view
  GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v30, v31);
  // pto: %out__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %out__ssa_v0_view
  int64_t v36 = v12 * v15;
  // pto: %out__ssa_v0_view
  int64_t v37 = v35 * v36;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v33, v34, v35, v12, v15);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v37, v37, v36, v15, v14);
  // pto: %out__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v38, v39);
  // pto: %weight__ssa_v0
  Tile<TileType::Mat, half, 384, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Mat, half, 384, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v13, v15);
  // pto: %weight__ssa_v0
  uint64_t v42 = (uint64_t) v4;
  TASSIGN(v41, v42);
  // pto: %w__ssa_v0_pview
  __gm__ half* v43 = PTOAS__GLOBAL_TENSOR_DATA(v32);
  // pto: %w__ssa_v0_pview
  pto::Shape<1, 1, 1, 384, 16> v44 = pto::Shape<1, 1, 1, 384, 16>();
  // pto: %w__ssa_v0_pview
  pto::Stride<6144, 6144, 6144, 16, 1> v45 = pto::Stride<6144, 6144, 6144, 16, 1>();
  // pto: %w__ssa_v0_pview
  GlobalTensor<half, pto::Shape<1, 1, 1, 384, 16>, pto::Stride<6144, 6144, 6144, 16, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 384, 16>, pto::Stride<6144, 6144, 6144, 16, 1>, pto::Layout::ND>(v43, v44, v45);
  TLOAD(v41, v46);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  // pto: %right__ssa_v0
  Tile<TileType::Right, half, 384, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Right, half, 384, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v13, v15);
  // pto: %right__ssa_v0
  uint64_t v48 = (uint64_t) v4;
  TASSIGN(v47, v48);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v47, v41);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  // pto: %pool__ssa_v0
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v12, v15);
  // pto: %pool__ssa_v0
  uint64_t v50 = (uint64_t) v4;
  TASSIGN(v49, v50);
  // pto: %values__ssa_v0
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v0
  uint64_t v52 = (uint64_t) v4;
  TASSIGN(v51, v52);
  // pto: %x__ssa_v0_pview
  __gm__ half* v53 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %x__ssa_v0_pview
  pto::Shape<1, 1, 1, 64, 384> v54 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %x__ssa_v0_pview
  pto::Stride<24576, 24576, 24576, 384, 1> v55 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %x__ssa_v0_pview
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v53, v54, v55);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  TLOAD(v51, v56);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  // pto: %left__ssa_v0
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v0
  uint64_t v58 = (uint64_t) v4;
  TASSIGN(v57, v58);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v57, v51);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  // pto: %part__ssa_v0
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v0
  uint64_t v60 = (uint64_t) v4;
  TASSIGN(v59, v60);
  // pto: %slice_view
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v61;
  // pto: %slice_view
  uint64_t v62 = (uint64_t) v4;
  TASSIGN(v61, v62);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v61, v57, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  // pto: %values__ssa_v1
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v1
  uint64_t v64 = (uint64_t) v4;
  TASSIGN(v63, v64);
  // pto: %0
  __gm__ half* v65 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %0
  unsigned v66 = 24576;
  // pto: %0
  pto::Shape<1, 1, 1, 64, 384> v67 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %0
  pto::Stride<24576, 24576, 24576, 384, 1> v68 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %0
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v69 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v65 + v66, v67, v68);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  TLOAD(v63, v69);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  // pto: %left__ssa_v1
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v1
  uint64_t v71 = (uint64_t) v4;
  TASSIGN(v70, v71);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  TMOV(v70, v63);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  // pto: %part__ssa_v1
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v1
  uint64_t v73 = (uint64_t) v5;
  TASSIGN(v72, v73);
  // pto: %1
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v74;
  // pto: %1
  uint64_t v75 = (uint64_t) v5;
  TASSIGN(v74, v75);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  TMATMUL(v74, v70, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  // pto: %values__ssa_v2
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v2
  uint64_t v77 = (uint64_t) v4;
  TASSIGN(v76, v77);
  // pto: %2
  __gm__ half* v78 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %2
  unsigned v79 = 49152;
  // pto: %2
  pto::Shape<1, 1, 1, 64, 384> v80 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %2
  pto::Stride<24576, 24576, 24576, 384, 1> v81 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %2
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v82 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v78 + v79, v80, v81);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  TLOAD(v76, v82);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
  // pto: %left__ssa_v2
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v2
  uint64_t v84 = (uint64_t) v4;
  TASSIGN(v83, v84);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  TMOV(v83, v76);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  // pto: %part__ssa_v2
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v2
  uint64_t v86 = (uint64_t) v6;
  TASSIGN(v85, v86);
  // pto: %3
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v87;
  // pto: %3
  uint64_t v88 = (uint64_t) v6;
  TASSIGN(v87, v88);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
  TMATMUL(v87, v83, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  // pto: %values__ssa_v3
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v3
  uint64_t v90 = (uint64_t) v4;
  TASSIGN(v89, v90);
  // pto: %4
  __gm__ half* v91 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %4
  unsigned v92 = 73728;
  // pto: %4
  pto::Shape<1, 1, 1, 64, 384> v93 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %4
  pto::Stride<24576, 24576, 24576, 384, 1> v94 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %4
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v95 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v91 + v92, v93, v94);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  TLOAD(v89, v95);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
  // pto: %left__ssa_v3
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v3
  uint64_t v97 = (uint64_t) v4;
  TASSIGN(v96, v97);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  TMOV(v96, v89);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  // pto: %part__ssa_v3
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v3
  uint64_t v99 = (uint64_t) v7;
  TASSIGN(v98, v99);
  // pto: %5
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v100;
  // pto: %5
  uint64_t v101 = (uint64_t) v7;
  TASSIGN(v100, v101);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
  TMATMUL(v100, v96, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  // pto: %values__ssa_v4
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v4
  uint64_t v103 = (uint64_t) v4;
  TASSIGN(v102, v103);
  // pto: %6
  __gm__ half* v104 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %6
  unsigned v105 = 98304;
  // pto: %6
  pto::Shape<1, 1, 1, 64, 384> v106 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %6
  pto::Stride<24576, 24576, 24576, 384, 1> v107 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %6
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v108 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v104 + v105, v106, v107);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  TLOAD(v102, v108);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
  // pto: %left__ssa_v4
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v4
  uint64_t v110 = (uint64_t) v4;
  TASSIGN(v109, v110);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  TMOV(v109, v102);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  // pto: %part__ssa_v4
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v4
  uint64_t v112 = (uint64_t) v8;
  TASSIGN(v111, v112);
  // pto: %7
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v113;
  // pto: %7
  uint64_t v114 = (uint64_t) v8;
  TASSIGN(v113, v114);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
  TMATMUL(v113, v109, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  // pto: %values__ssa_v5
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v5
  uint64_t v116 = (uint64_t) v4;
  TASSIGN(v115, v116);
  // pto: %8
  __gm__ half* v117 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %8
  unsigned v118 = 122880;
  // pto: %8
  pto::Shape<1, 1, 1, 64, 384> v119 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %8
  pto::Stride<24576, 24576, 24576, 384, 1> v120 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %8
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v121 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v117 + v118, v119, v120);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  TLOAD(v115, v121);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
  // pto: %left__ssa_v5
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v5
  uint64_t v123 = (uint64_t) v4;
  TASSIGN(v122, v123);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  TMOV(v122, v115);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  // pto: %part__ssa_v5
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v5
  uint64_t v125 = (uint64_t) v9;
  TASSIGN(v124, v125);
  // pto: %9
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v126;
  // pto: %9
  uint64_t v127 = (uint64_t) v9;
  TASSIGN(v126, v127);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
  TMATMUL(v126, v122, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  // pto: %values__ssa_v6
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v6
  uint64_t v129 = (uint64_t) v4;
  TASSIGN(v128, v129);
  // pto: %10
  __gm__ half* v130 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %10
  unsigned v131 = 147456;
  // pto: %10
  pto::Shape<1, 1, 1, 64, 384> v132 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %10
  pto::Stride<24576, 24576, 24576, 384, 1> v133 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %10
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v134 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v130 + v131, v132, v133);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  TLOAD(v128, v134);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  // pto: %left__ssa_v6
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v135 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v6
  uint64_t v136 = (uint64_t) v4;
  TASSIGN(v135, v136);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  TMOV(v135, v128);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  // pto: %part__ssa_v6
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v6
  uint64_t v138 = (uint64_t) v10;
  TASSIGN(v137, v138);
  // pto: %11
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v139;
  // pto: %11
  uint64_t v140 = (uint64_t) v10;
  TASSIGN(v139, v140);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
  TMATMUL(v139, v135, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  // pto: %values__ssa_v7
  Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Mat, half, 64, 384, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %values__ssa_v7
  uint64_t v142 = (uint64_t) v4;
  TASSIGN(v141, v142);
  // pto: %12
  __gm__ half* v143 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %12
  unsigned v144 = 172032;
  // pto: %12
  pto::Shape<1, 1, 1, 64, 384> v145 = pto::Shape<1, 1, 1, 64, 384>();
  // pto: %12
  pto::Stride<24576, 24576, 24576, 384, 1> v146 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %12
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v147 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 384>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v143 + v144, v145, v146);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  TLOAD(v141, v147);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  // pto: %left__ssa_v7
  Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Left, half, 64, 384, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v16, v13);
  // pto: %left__ssa_v7
  uint64_t v149 = (uint64_t) v4;
  TASSIGN(v148, v149);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  TMOV(v148, v141);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  // pto: %part__ssa_v7
  Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Acc, float, 64, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %part__ssa_v7
  uint64_t v151 = (uint64_t) v11;
  TASSIGN(v150, v151);
  // pto: %13
  Tile<TileType::Acc, float, 512, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v152;
  // pto: %13
  uint64_t v153 = (uint64_t) v11;
  TASSIGN(v152, v153);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  TMATMUL(v152, v148, v47);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %out__ssa_v0_pview
  __gm__ float* v154 = PTOAS__GLOBAL_TENSOR_DATA(v40);
  // pto: %out__ssa_v0_pview
  pto::Shape<1, 1, 1, 512, 16> v155 = pto::Shape<1, 1, 1, 512, 16>();
  // pto: %out__ssa_v0_pview
  pto::Stride<8192, 8192, 8192, 16, 1> v156 = pto::Stride<8192, 8192, 8192, 16, 1>();
  // pto: %out__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 512, 16>, pto::Stride<8192, 8192, 8192, 16, 1>, pto::Layout::ND> v157 = GlobalTensor<float, pto::Shape<1, 1, 1, 512, 16>, pto::Stride<8192, 8192, 8192, 16, 1>, pto::Layout::ND>(v154, v155, v156);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v157, v49);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}