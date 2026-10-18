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

AICORE void boxed_disjoint(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  const int64_t v5 = 32768;
  // pto: %c1
  const int64_t v6 = 1;
  // pto: %c64
  const int64_t v7 = 64;
  // pto: %c128
  const int64_t v8 = 128;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %a16384
  const int64_t v10 = 16384;
  // pto: %tva
  const int64_t v11 = 1;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  int64_t v14 = v7 * v8;
  // pto: %tva
  int64_t v15 = v13 * v14;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v16 = pto::Shape<1, 1, 1, -1, -1>(v11, v12, v13, v7, v8);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v17 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v15, v15, v14, v8, v6);
  // pto: %tva
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v18 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v16, v17);
  // pto: %va
  __gm__ bfloat16_t* v19 = PTOAS__GLOBAL_TENSOR_DATA(v18);
  // pto: %va
  pto::Shape<1, 1, 1, 64, 128> v20 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %va
  pto::Stride<8192, 8192, 8192, 128, 1> v21 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %va
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v22 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v19, v20, v21);
  // pto: %tvb
  const int64_t v23 = 1;
  // pto: %tvb
  const int64_t v24 = 1;
  // pto: %tvb
  const int64_t v25 = 1;
  // pto: %tvb
  int64_t v26 = v8 * v8;
  // pto: %tvb
  int64_t v27 = v25 * v26;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v23, v24, v25, v8, v8);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v27, v27, v26, v8, v6);
  // pto: %tvb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v28, v29);
  // pto: %vb
  __gm__ bfloat16_t* v31 = PTOAS__GLOBAL_TENSOR_DATA(v30);
  // pto: %vb
  pto::Shape<1, 1, 1, 128, 128> v32 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %vb
  pto::Stride<16384, 16384, 16384, 128, 1> v33 = pto::Stride<16384, 16384, 16384, 128, 1>();
  // pto: %vb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v34 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v31, v32, v33);
  // pto: %tvo0
  const int64_t v35 = 1;
  // pto: %tvo0
  const int64_t v36 = 1;
  // pto: %tvo0
  const int64_t v37 = 1;
  // pto: %tvo0
  int64_t v38 = v7 * v8;
  // pto: %tvo0
  int64_t v39 = v37 * v38;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v35, v36, v37, v7, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v39, v39, v38, v8, v6);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v40, v41);
  // pto: %vo0
  __gm__ float* v43 = PTOAS__GLOBAL_TENSOR_DATA(v42);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v44 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<8192, 8192, 8192, 128, 1> v45 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v43, v44, v45);
  // pto: %tvo1
  const int64_t v47 = 1;
  // pto: %tvo1
  const int64_t v48 = 1;
  // pto: %tvo1
  const int64_t v49 = 1;
  // pto: %tvo1
  int64_t v50 = v7 * v8;
  // pto: %tvo1
  int64_t v51 = v49 * v50;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v52 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v7, v8);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v53 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v51, v51, v50, v8, v6);
  // pto: %tvo1
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v52, v53);
  // pto: %vo1
  __gm__ float* v55 = PTOAS__GLOBAL_TENSOR_DATA(v54);
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v56 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<8192, 8192, 8192, 128, 1> v57 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v55, v56, v57);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v59;
  // pto: %am
  uint64_t v60 = (uint64_t) v9;
  TASSIGN(v59, v60);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v61;
  // pto: %bm
  uint64_t v62 = (uint64_t) v10;
  TASSIGN(v61, v62);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v63;
  // pto: %al
  uint64_t v64 = (uint64_t) v9;
  TASSIGN(v63, v64);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v65;
  // pto: %br
  uint64_t v66 = (uint64_t) v9;
  TASSIGN(v65, v66);
  // pto: %pool
  uint64_t v67 = (uint64_t) v9;
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v68;
  // pto: %cfirst
  uint64_t v69 = (uint64_t) v9;
  TASSIGN(v68, v69);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v70;
  // pto: %csecond
  uint64_t v71 = (uint64_t) v5;
  TASSIGN(v70, v71);
  TLOAD(v59, v22);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v61, v34);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v63, v59);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v65, v61);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v68, v63, v65);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v46, v68);
  TMATMUL(v70, v63, v65);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v58, v70);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}