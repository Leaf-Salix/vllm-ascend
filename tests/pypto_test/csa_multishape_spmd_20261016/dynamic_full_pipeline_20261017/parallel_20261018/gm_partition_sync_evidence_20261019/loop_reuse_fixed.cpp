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

AICORE void loop_reuse(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5) {
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
  int64_t v41 = v8 * v10;
  // pto: %tvo0
  int64_t v42 = v40 * v41;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v38, v39, v40, v8, v10);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v42, v42, v41, v10, v7);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v43, v44);
  // pto: %vo0
  __gm__ float* v46 = PTOAS__GLOBAL_TENSOR_DATA(v45);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v47 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v48 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v46, v47, v48);
  // pto: %vo1
  __gm__ float* v50 = PTOAS__GLOBAL_TENSOR_DATA(v45);
  // pto: %vo1
  unsigned v51 = 128;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v52 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v53 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v50 + v51, v52, v53);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55;
  // pto: %am
  uint64_t v56 = (uint64_t) v11;
  TASSIGN(v55, v56);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57;
  // pto: %bm
  uint64_t v58 = (uint64_t) v12;
  TASSIGN(v57, v58);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v59;
  // pto: %al
  uint64_t v60 = (uint64_t) v11;
  TASSIGN(v59, v60);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v61;
  // pto: %br
  uint64_t v62 = (uint64_t) v11;
  TASSIGN(v61, v62);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v63;
  // pto: %cfirst
  uint64_t v64 = (uint64_t) v11;
  TASSIGN(v63, v64);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v65;
  // pto: %csecond
  uint64_t v66 = (uint64_t) v13;
  TASSIGN(v65, v66);
  TLOAD(v55, v25);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v57, v37);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v59, v55);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v61, v57);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v63, v59, v61);
  TMATMUL(v65, v59, v61);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  for (int64_t v67 = v6; v67 < v5; v67 += v7) {
    pipe_barrier(PIPE_FIX);
    TSTORE(v49, v63);
    TSTORE(v54, v65);
  }
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}