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

AICORE void boxed_overlap(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
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
  // pto: %pool
  uint64_t v66 = (uint64_t) v8;
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v67;
  // pto: %cfirst
  uint64_t v68 = (uint64_t) v8;
  TASSIGN(v67, v68);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v69;
  // pto: %csecond
  uint64_t v70 = (uint64_t) v8;
  TASSIGN(v69, v70);
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
  TMATMUL(v67, v62, v64);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v45, v67);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL(v69, v62, v64);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v57, v69);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void separate_disjoint(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
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
  // pto: %a32768
  const int64_t v10 = 32768;
  // pto: %tva
  const int64_t v11 = 1;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  int64_t v14 = v6 * v7;
  // pto: %tva
  int64_t v15 = v13 * v14;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v16 = pto::Shape<1, 1, 1, -1, -1>(v11, v12, v13, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v17 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v15, v15, v14, v7, v5);
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
  int64_t v26 = v7 * v7;
  // pto: %tvb
  int64_t v27 = v25 * v26;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v23, v24, v25, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v27, v27, v26, v7, v5);
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
  int64_t v38 = v6 * v7;
  // pto: %tvo0
  int64_t v39 = v37 * v38;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v35, v36, v37, v6, v7);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v39, v39, v38, v7, v5);
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
  int64_t v50 = v6 * v7;
  // pto: %tvo1
  int64_t v51 = v49 * v50;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v52 = pto::Shape<1, 1, 1, -1, -1>(v47, v48, v49, v6, v7);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v53 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v51, v51, v50, v7, v5);
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
  uint64_t v60 = (uint64_t) v8;
  TASSIGN(v59, v60);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v61;
  // pto: %bm
  uint64_t v62 = (uint64_t) v9;
  TASSIGN(v61, v62);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v63;
  // pto: %al
  uint64_t v64 = (uint64_t) v8;
  TASSIGN(v63, v64);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v65;
  // pto: %br
  uint64_t v66 = (uint64_t) v8;
  TASSIGN(v65, v66);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v67;
  // pto: %cfirst
  uint64_t v68 = (uint64_t) v8;
  TASSIGN(v67, v68);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v69;
  // pto: %csecond
  uint64_t v70 = (uint64_t) v10;
  TASSIGN(v69, v70);
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
  TMATMUL(v67, v63, v65);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v46, v67);
  TMATMUL(v69, v63, v65);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v58, v69);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void boxed_partial_overlap(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %a0
  const int64_t v8 = 0;
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
  // pto: %pool
  uint64_t v66 = (uint64_t) v8;
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v67;
  // pto: %cfirst
  uint64_t v68 = (uint64_t) v8;
  TASSIGN(v67, v68);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v69;
  // pto: %csecond
  uint64_t v70 = (uint64_t) v9;
  TASSIGN(v69, v70);
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
  TMATMUL(v67, v62, v64);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v45, v67);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL(v69, v62, v64);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v57, v69);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void boxed_dynamic(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5) {
  using T = float;
  const int64_t v6 = 256;
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
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v69;
  // pto: %cfirst
  uint64_t v70 = (uint64_t) v10;
  TASSIGN(v69, v70);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v71;
  // pto: %csecond
  uint64_t v72 = (uint64_t) ((int64_t) (uint64_t) v5 * (uint64_t) v6);
  TASSIGN(v71, v72);
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
  TMATMUL(v69, v64, v66);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v69);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL(v71, v64, v66);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v59, v71);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
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