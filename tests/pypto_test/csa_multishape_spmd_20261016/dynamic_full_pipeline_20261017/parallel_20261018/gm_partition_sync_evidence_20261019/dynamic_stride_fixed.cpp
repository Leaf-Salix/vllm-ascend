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

AICORE void dynamic_stride(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5) {
  using T = float;
  // pto: %c0
  const int64_t v6 = 0;
  // pto: %c1
  const int64_t v7 = 1;
  // pto: %c64
  const int64_t v8 = 64;
  // pto: %c128
  const int64_t v9 = 128;
  // pto: %c256
  const int64_t v10 = 256;
  // pto: %a0
  const int64_t v11 = 0;
  // pto: %a16384
  const int64_t v12 = 16384;
  // pto: %a32768
  const int64_t v13 = 32768;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  const int64_t v15 = 1;
  // pto: %tva
  const int64_t v16 = 1;
  // pto: %tva
  int64_t v17 = v8 * v9;
  // pto: %tva
  int64_t v18 = v16 * v17;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v19 = pto::Shape<1, 1, 1, -1, -1>(v14, v15, v16, v8, v9);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v20 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v18, v18, v17, v9, v7);
  // pto: %tva
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v21 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v19, v20);
  // pto: %va
  __gm__ bfloat16_t* v22 = PTOAS__GLOBAL_TENSOR_DATA(v21);
  // pto: %va
  pto::Shape<1, 1, 1, 64, 128> v23 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %va
  pto::Stride<8192, 8192, 8192, 128, 1> v24 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %va
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v25 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v22, v23, v24);
  // pto: %tvb
  const int64_t v26 = 1;
  // pto: %tvb
  const int64_t v27 = 1;
  // pto: %tvb
  const int64_t v28 = 1;
  // pto: %tvb
  int64_t v29 = v9 * v9;
  // pto: %tvb
  int64_t v30 = v28 * v29;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v31 = pto::Shape<1, 1, 1, -1, -1>(v26, v27, v28, v9, v9);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v32 = pto::Stride<-1, -1, -1, -1, -1>(v27 * v30, v30, v29, v9, v7);
  // pto: %tvb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v33 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v31, v32);
  // pto: %vb
  __gm__ bfloat16_t* v34 = PTOAS__GLOBAL_TENSOR_DATA(v33);
  // pto: %vb
  pto::Shape<1, 1, 1, 128, 128> v35 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %vb
  pto::Stride<16384, 16384, 16384, 128, 1> v36 = pto::Stride<16384, 16384, 16384, 128, 1>();
  // pto: %vb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v34, v35, v36);
  // pto: %tvo0
  const int64_t v38 = 1;
  // pto: %tvo0
  const int64_t v39 = 1;
  // pto: %tvo0
  const int64_t v40 = 1;
  // pto: %tvo0
  int64_t v41 = (int64_t) v5;
  // pto: %tvo0
  int64_t v42 = v8 * v41;
  // pto: %tvo0
  int64_t v43 = v40 * v42;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v8, v10);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v43, v43, v42, v41, v7);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v44, v45);
  // pto: %vo0
  int64_t v47 = (int64_t) v5;
  // pto: %vo0
  const int64_t v48 = 0;
  // pto: %vo0
  __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v46);
  // pto: %vo0
  const int64_t v50 = 1;
  // pto: %vo0
  const int64_t v51 = 1;
  // pto: %vo0
  const int64_t v52 = 1;
  // pto: %vo0
  int64_t v53 = v8 * v47;
  // pto: %vo0
  int64_t v54 = v52 * v53;
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v55 = pto::Shape<1, 1, 1, 64, 128>(v50, v51, v52, v8, v9);
  // pto: %vo0
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v51 * v54, v54, v53, v47, v7);
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v49 + (v48 + v6 * v47 + v6 * v7), v55, v56);
  // pto: %vo1
  int64_t v58 = (int64_t) v5;
  // pto: %vo1
  const int64_t v59 = 0;
  // pto: %vo1
  __gm__ float* v60 = PTOAS__GLOBAL_TENSOR_DATA(v46);
  // pto: %vo1
  const int64_t v61 = 1;
  // pto: %vo1
  const int64_t v62 = 1;
  // pto: %vo1
  const int64_t v63 = 1;
  // pto: %vo1
  int64_t v64 = v8 * v58;
  // pto: %vo1
  int64_t v65 = v63 * v64;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v66 = pto::Shape<1, 1, 1, 64, 128>(v61, v62, v63, v8, v9);
  // pto: %vo1
  pto::Stride<-1, -1, -1, -1, -1> v67 = pto::Stride<-1, -1, -1, -1, -1>(v62 * v65, v65, v64, v58, v7);
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v68 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v60 + (v59 + v6 * v58 + v9 * v7), v66, v67);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v69;
  // pto: %am
  uint64_t v70 = (uint64_t) v11;
  TASSIGN(v69, v70);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v71;
  // pto: %bm
  uint64_t v72 = (uint64_t) v12;
  TASSIGN(v71, v72);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v73;
  // pto: %al
  uint64_t v74 = (uint64_t) v11;
  TASSIGN(v73, v74);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v75;
  // pto: %br
  uint64_t v76 = (uint64_t) v11;
  TASSIGN(v75, v76);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v77;
  // pto: %cfirst
  uint64_t v78 = (uint64_t) v11;
  TASSIGN(v77, v78);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v79;
  // pto: %csecond
  uint64_t v80 = (uint64_t) v13;
  TASSIGN(v79, v80);
  TLOAD(v69, v25);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v71, v37);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v73, v69);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v75, v71);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v77, v73, v75);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v57, v77);
  TMATMUL(v79, v73, v75);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v68, v79);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}