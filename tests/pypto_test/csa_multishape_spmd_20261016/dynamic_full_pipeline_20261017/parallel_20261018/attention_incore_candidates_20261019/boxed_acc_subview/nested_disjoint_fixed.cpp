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

AICORE void nested_disjoint(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  const int64_t v5 = 65536;
  const int64_t v6 = 32768;
  // pto: %c1
  const int64_t v7 = 1;
  // pto: %c64
  const int64_t v8 = 64;
  // pto: %c128
  const int64_t v9 = 128;
  // pto: %a0
  const int64_t v10 = 0;
  // pto: %a16384
  const int64_t v11 = 16384;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  int64_t v15 = v8 * v9;
  // pto: %tva
  int64_t v16 = v14 * v15;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v8, v9);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v9, v7);
  // pto: %tva
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v17, v18);
  // pto: %va
  __gm__ bfloat16_t* v20 = PTOAS__GLOBAL_TENSOR_DATA(v19);
  // pto: %va
  pto::Shape<1, 1, 1, 64, 128> v21 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %va
  pto::Stride<8192, 8192, 8192, 128, 1> v22 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %va
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v23 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v20, v21, v22);
  // pto: %tvb
  const int64_t v24 = 1;
  // pto: %tvb
  const int64_t v25 = 1;
  // pto: %tvb
  const int64_t v26 = 1;
  // pto: %tvb
  int64_t v27 = v9 * v9;
  // pto: %tvb
  int64_t v28 = v26 * v27;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v9, v9);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v9, v7);
  // pto: %tvb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v31 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v29, v30);
  // pto: %vb
  __gm__ bfloat16_t* v32 = PTOAS__GLOBAL_TENSOR_DATA(v31);
  // pto: %vb
  pto::Shape<1, 1, 1, 128, 128> v33 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %vb
  pto::Stride<16384, 16384, 16384, 128, 1> v34 = pto::Stride<16384, 16384, 16384, 128, 1>();
  // pto: %vb
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v32, v33, v34);
  // pto: %tvo0
  const int64_t v36 = 1;
  // pto: %tvo0
  const int64_t v37 = 1;
  // pto: %tvo0
  const int64_t v38 = 1;
  // pto: %tvo0
  int64_t v39 = v8 * v9;
  // pto: %tvo0
  int64_t v40 = v38 * v39;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v8, v9);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v9, v7);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v41, v42);
  // pto: %vo0
  __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v45 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<8192, 8192, 8192, 128, 1> v46 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v44, v45, v46);
  // pto: %tvo1
  const int64_t v48 = 1;
  // pto: %tvo1
  const int64_t v49 = 1;
  // pto: %tvo1
  const int64_t v50 = 1;
  // pto: %tvo1
  int64_t v51 = v8 * v9;
  // pto: %tvo1
  int64_t v52 = v50 * v51;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v48, v49, v50, v8, v9);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v49 * v52, v52, v51, v9, v7);
  // pto: %tvo1
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v53, v54);
  // pto: %vo1
  __gm__ float* v56 = PTOAS__GLOBAL_TENSOR_DATA(v55);
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v57 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<8192, 8192, 8192, 128, 1> v58 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v56, v57, v58);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v60;
  // pto: %am
  uint64_t v61 = (uint64_t) v10;
  TASSIGN(v60, v61);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v62;
  // pto: %bm
  uint64_t v63 = (uint64_t) v11;
  TASSIGN(v62, v63);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v64;
  // pto: %al
  uint64_t v65 = (uint64_t) v10;
  TASSIGN(v64, v65);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v66;
  // pto: %br
  uint64_t v67 = (uint64_t) v10;
  TASSIGN(v66, v67);
  // pto: %pool
  uint64_t v68 = (uint64_t) v10;
  // pto: %parent
  uint64_t v69 = (uint64_t) v6;
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v70;
  // pto: %cfirst
  uint64_t v71 = (uint64_t) v6;
  TASSIGN(v70, v71);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v72;
  // pto: %csecond
  uint64_t v73 = (uint64_t) v5;
  TASSIGN(v72, v73);
  TLOAD(v60, v23);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v62, v35);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v64, v60);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v66, v62);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v70, v64, v66);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v70);
  TMATMUL(v72, v64, v66);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v59, v72);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}