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

AICORE void unknown_pointer_offset(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5) {
  using T = float;
  // pto: %c1
  const int64_t v6 = 1;
  // pto: %c64
  const int64_t v7 = 64;
  // pto: %c128
  const int64_t v8 = 128;
  // pto: %c256
  const int64_t v9 = 256;
  // pto: %a0
  const int64_t v10 = 0;
  // pto: %a16384
  const int64_t v11 = 16384;
  // pto: %a32768
  const int64_t v12 = 32768;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  const int64_t v15 = 1;
  // pto: %tva
  int64_t v16 = v7 * v8;
  // pto: %tva
  int64_t v17 = v15 * v16;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v13, v14, v15, v7, v8);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v14 * v17, v17, v16, v8, v6);
  // pto: %tva
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v20 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v18, v19);
  // pto: %va
  __gm__ bfloat16_t* v21 = PTOAS__GLOBAL_TENSOR_DATA(v20);
  // pto: %va
  pto::Shape<1, 1, 1, 64, 128> v22 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %va
  pto::Stride<8192, 8192, 8192, 128, 1> v23 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %va
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v24 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v21, v22, v23);
  // pto: %tvb
  const int64_t v25 = 1;
  // pto: %tvb
  const int64_t v26 = 1;
  // pto: %tvb
  const int64_t v27 = 1;
  // pto: %tvb
  int64_t v28 = v8 * v8;
  // pto: %tvb
  int64_t v29 = v27 * v28;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v8, v8);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v8, v6);
  // pto: %tvb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v30, v31);
  // pto: %vb
  __gm__ bfloat16_t* v33 = PTOAS__GLOBAL_TENSOR_DATA(v32);
  // pto: %vb
  pto::Shape<1, 1, 1, 128, 128> v34 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %vb
  pto::Stride<16384, 16384, 16384, 128, 1> v35 = pto::Stride<16384, 16384, 16384, 128, 1>();
  // pto: %vb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v36 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v33, v34, v35);
  // pto: %tvo0
  const int64_t v37 = 1;
  // pto: %tvo0
  const int64_t v38 = 1;
  // pto: %tvo0
  const int64_t v39 = 1;
  // pto: %tvo0
  int64_t v40 = v7 * v9;
  // pto: %tvo0
  int64_t v41 = v39 * v40;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v37, v38, v39, v7, v9);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v9, v6);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v42, v43);
  // pto: %vo0
  __gm__ float* v45 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v46 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v47 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v45, v46, v47);
  // pto: %tvo1
  const int64_t v49 = 1;
  // pto: %tvo1
  const int64_t v50 = 1;
  // pto: %tvo1
  const int64_t v51 = 1;
  // pto: %tvo1
  int64_t v52 = v7 * v9;
  // pto: %tvo1
  int64_t v53 = v51 * v52;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v7, v9);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v9, v6);
  // pto: %shifted, %tvo1
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3 + v5, v54, v55);
  // pto: %vo1
  __gm__ float* v57 = PTOAS__GLOBAL_TENSOR_DATA(v56);
  // pto: %vo1
  unsigned v58 = 128;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v59 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v60 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v57 + v58, v59, v60);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v62;
  // pto: %am
  uint64_t v63 = (uint64_t) v10;
  TASSIGN(v62, v63);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v64;
  // pto: %bm
  uint64_t v65 = (uint64_t) v11;
  TASSIGN(v64, v65);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v66;
  // pto: %al
  uint64_t v67 = (uint64_t) v10;
  TASSIGN(v66, v67);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v68;
  // pto: %br
  uint64_t v69 = (uint64_t) v10;
  TASSIGN(v68, v69);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v70;
  // pto: %cfirst
  uint64_t v71 = (uint64_t) v10;
  TASSIGN(v70, v71);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v72;
  // pto: %csecond
  uint64_t v73 = (uint64_t) v12;
  TASSIGN(v72, v73);
  TLOAD(v62, v24);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v64, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v66, v62);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v68, v64);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v70, v66, v68);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v48, v70);
  TMATMUL(v72, v66, v68);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v61, v72);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}