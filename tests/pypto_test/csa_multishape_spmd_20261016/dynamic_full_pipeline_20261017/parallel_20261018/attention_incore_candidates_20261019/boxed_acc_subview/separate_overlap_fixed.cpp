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

AICORE void separate_overlap(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %a0
  const int64_t v8 = 0;
  // pto: %a16384
  const int64_t v9 = 16384;
  // pto: %tva
  const int64_t v10 = 1;
  // pto: %tva
  const int64_t v11 = 1;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  int64_t v13 = v6 * v7;
  // pto: %tva
  int64_t v14 = v12 * v13;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v10, v11, v12, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v14, v14, v13, v7, v5);
  // pto: %tva
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v17 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v15, v16);
  // pto: %va
  __gm__ bfloat16_t* v18 = PTOAS__GLOBAL_TENSOR_DATA(v17);
  // pto: %va
  pto::Shape<1, 1, 1, 64, 128> v19 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %va
  pto::Stride<8192, 8192, 8192, 128, 1> v20 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %va
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v21 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v18, v19, v20);
  // pto: %tvb
  const int64_t v22 = 1;
  // pto: %tvb
  const int64_t v23 = 1;
  // pto: %tvb
  const int64_t v24 = 1;
  // pto: %tvb
  int64_t v25 = v7 * v7;
  // pto: %tvb
  int64_t v26 = v24 * v25;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v7, v5);
  // pto: %tvb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v27, v28);
  // pto: %vb
  __gm__ bfloat16_t* v30 = PTOAS__GLOBAL_TENSOR_DATA(v29);
  // pto: %vb
  pto::Shape<1, 1, 1, 128, 128> v31 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %vb
  pto::Stride<16384, 16384, 16384, 128, 1> v32 = pto::Stride<16384, 16384, 16384, 128, 1>();
  // pto: %vb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v33 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v30, v31, v32);
  // pto: %tvo0
  const int64_t v34 = 1;
  // pto: %tvo0
  const int64_t v35 = 1;
  // pto: %tvo0
  const int64_t v36 = 1;
  // pto: %tvo0
  int64_t v37 = v6 * v7;
  // pto: %tvo0
  int64_t v38 = v36 * v37;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v39 = pto::Shape<1, 1, 1, -1, -1>(v34, v35, v36, v6, v7);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v40 = pto::Stride<-1, -1, -1, -1, -1>(v35 * v38, v38, v37, v7, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v39, v40);
  // pto: %vo0
  __gm__ float* v42 = PTOAS__GLOBAL_TENSOR_DATA(v41);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v43 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<8192, 8192, 8192, 128, 1> v44 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v42, v43, v44);
  // pto: %tvo1
  const int64_t v46 = 1;
  // pto: %tvo1
  const int64_t v47 = 1;
  // pto: %tvo1
  const int64_t v48 = 1;
  // pto: %tvo1
  int64_t v49 = v6 * v7;
  // pto: %tvo1
  int64_t v50 = v48 * v49;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v51 = pto::Shape<1, 1, 1, -1, -1>(v46, v47, v48, v6, v7);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v50, v50, v49, v7, v5);
  // pto: %tvo1
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v51, v52);
  // pto: %vo1
  __gm__ float* v54 = PTOAS__GLOBAL_TENSOR_DATA(v53);
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v55 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<8192, 8192, 8192, 128, 1> v56 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v54, v55, v56);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v58;
  // pto: %am
  uint64_t v59 = (uint64_t) v8;
  TASSIGN(v58, v59);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v60;
  // pto: %bm
  uint64_t v61 = (uint64_t) v9;
  TASSIGN(v60, v61);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v62;
  // pto: %al
  uint64_t v63 = (uint64_t) v8;
  TASSIGN(v62, v63);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v64;
  // pto: %br
  uint64_t v65 = (uint64_t) v8;
  TASSIGN(v64, v65);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v66;
  // pto: %cfirst
  uint64_t v67 = (uint64_t) v8;
  TASSIGN(v66, v67);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v68;
  // pto: %csecond
  uint64_t v69 = (uint64_t) v8;
  TASSIGN(v68, v69);
  TLOAD(v58, v21);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v60, v33);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v62, v58);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v64, v60);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v66, v62, v64);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v45, v66);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL(v68, v62, v64);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v57, v68);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}