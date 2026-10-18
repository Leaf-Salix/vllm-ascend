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

AICORE void separate_roots(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %a16384
  const int64_t v10 = 16384;
  // pto: %a32768
  const int64_t v11 = 32768;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  int64_t v15 = v6 * v7;
  // pto: %tva
  int64_t v16 = v14 * v15;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v7, v5);
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
  int64_t v27 = v7 * v7;
  // pto: %tvb
  int64_t v28 = v26 * v27;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v7, v5);
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
  int64_t v39 = v6 * v8;
  // pto: %tvo0
  int64_t v40 = v38 * v39;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v6, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v8, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v41, v42);
  // pto: %vo0
  __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v45 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v46 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v44, v45, v46);
  // pto: %tvo1
  const int64_t v48 = 1;
  // pto: %tvo1
  const int64_t v49 = 1;
  // pto: %tvo1
  const int64_t v50 = 1;
  // pto: %tvo1
  int64_t v51 = v6 * v8;
  // pto: %tvo1
  int64_t v52 = v50 * v51;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v48, v49, v50, v6, v8);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v49 * v52, v52, v51, v8, v5);
  // pto: %tvo1
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v53, v54);
  // pto: %vo1
  __gm__ float* v56 = PTOAS__GLOBAL_TENSOR_DATA(v55);
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v57 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v58 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v56, v57, v58);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v60;
  // pto: %am
  uint64_t v61 = (uint64_t) v9;
  TASSIGN(v60, v61);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v62;
  // pto: %bm
  uint64_t v63 = (uint64_t) v10;
  TASSIGN(v62, v63);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v64;
  // pto: %al
  uint64_t v65 = (uint64_t) v9;
  TASSIGN(v64, v65);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v66;
  // pto: %br
  uint64_t v67 = (uint64_t) v9;
  TASSIGN(v66, v67);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v68;
  // pto: %cfirst
  uint64_t v69 = (uint64_t) v9;
  TASSIGN(v68, v69);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v70;
  // pto: %csecond
  uint64_t v71 = (uint64_t) v11;
  TASSIGN(v70, v71);
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
  TMATMUL(v68, v64, v66);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v68);
  TMATMUL(v70, v64, v66);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v59, v70);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void same_root_disjoint(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %a16384
  const int64_t v10 = 16384;
  // pto: %a32768
  const int64_t v11 = 32768;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  int64_t v15 = v6 * v7;
  // pto: %tva
  int64_t v16 = v14 * v15;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v7, v5);
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
  int64_t v27 = v7 * v7;
  // pto: %tvb
  int64_t v28 = v26 * v27;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v7, v5);
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
  int64_t v39 = v6 * v8;
  // pto: %tvo0
  int64_t v40 = v38 * v39;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v6, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v8, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v41, v42);
  // pto: %vo0
  __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v45 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v46 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v44, v45, v46);
  // pto: %vo1
  __gm__ float* v48 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo1
  unsigned v49 = 128;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v50 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v51 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v52 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v48 + v49, v50, v51);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v53;
  // pto: %am
  uint64_t v54 = (uint64_t) v9;
  TASSIGN(v53, v54);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55;
  // pto: %bm
  uint64_t v56 = (uint64_t) v10;
  TASSIGN(v55, v56);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57;
  // pto: %al
  uint64_t v58 = (uint64_t) v9;
  TASSIGN(v57, v58);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v59;
  // pto: %br
  uint64_t v60 = (uint64_t) v9;
  TASSIGN(v59, v60);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v61;
  // pto: %cfirst
  uint64_t v62 = (uint64_t) v9;
  TASSIGN(v61, v62);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v63;
  // pto: %csecond
  uint64_t v64 = (uint64_t) v11;
  TASSIGN(v63, v64);
  TLOAD(v53, v23);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v55, v35);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v53);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v59, v55);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v61, v57, v59);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v61);
  TMATMUL(v63, v57, v59);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v52, v63);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void same_root_overlap(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %a16384
  const int64_t v10 = 16384;
  // pto: %a32768
  const int64_t v11 = 32768;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  int64_t v15 = v6 * v7;
  // pto: %tva
  int64_t v16 = v14 * v15;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v7, v5);
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
  int64_t v27 = v7 * v7;
  // pto: %tvb
  int64_t v28 = v26 * v27;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v7, v5);
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
  int64_t v39 = v6 * v8;
  // pto: %tvo0
  int64_t v40 = v38 * v39;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v6, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v8, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v41, v42);
  // pto: %vo0
  __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v45 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v46 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v44, v45, v46);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v48;
  // pto: %am
  uint64_t v49 = (uint64_t) v9;
  TASSIGN(v48, v49);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v50;
  // pto: %bm
  uint64_t v51 = (uint64_t) v10;
  TASSIGN(v50, v51);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v52;
  // pto: %al
  uint64_t v53 = (uint64_t) v9;
  TASSIGN(v52, v53);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v54;
  // pto: %br
  uint64_t v55 = (uint64_t) v9;
  TASSIGN(v54, v55);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v56;
  // pto: %cfirst
  uint64_t v57 = (uint64_t) v9;
  TASSIGN(v56, v57);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v58;
  // pto: %csecond
  uint64_t v59 = (uint64_t) v11;
  TASSIGN(v58, v59);
  TLOAD(v48, v23);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v50, v35);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v52, v48);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v54, v50);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v56, v52, v54);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v56);
  TMATMUL(v58, v52, v54);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v47, v58);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void dynamic_row_disjoint(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5) {
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
  // pto: %vo1
  __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo1
  const int64_t v50 = 128;
  // pto: %vo1
  const int64_t v51 = 256;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v52 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v53 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v49 + (v50 + v5 * v51), v52, v53);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55;
  // pto: %am
  uint64_t v56 = (uint64_t) v10;
  TASSIGN(v55, v56);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57;
  // pto: %bm
  uint64_t v58 = (uint64_t) v11;
  TASSIGN(v57, v58);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v59;
  // pto: %al
  uint64_t v60 = (uint64_t) v10;
  TASSIGN(v59, v60);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v61;
  // pto: %br
  uint64_t v62 = (uint64_t) v10;
  TASSIGN(v61, v62);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v63;
  // pto: %cfirst
  uint64_t v64 = (uint64_t) v10;
  TASSIGN(v63, v64);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v65;
  // pto: %csecond
  uint64_t v66 = (uint64_t) v12;
  TASSIGN(v65, v66);
  TLOAD(v55, v24);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v57, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v59, v55);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v61, v57);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v63, v59, v61);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v48, v63);
  TMATMUL(v65, v59, v61);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(v54, v65);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void dynamic_column_unknown(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5) {
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
  // pto: %vo1
  __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo1
  const int64_t v50 = 0;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v51 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v52 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v49 + (v50 + v5), v51, v52);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v54;
  // pto: %am
  uint64_t v55 = (uint64_t) v10;
  TASSIGN(v54, v55);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v56;
  // pto: %bm
  uint64_t v57 = (uint64_t) v11;
  TASSIGN(v56, v57);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v58;
  // pto: %al
  uint64_t v59 = (uint64_t) v10;
  TASSIGN(v58, v59);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v60;
  // pto: %br
  uint64_t v61 = (uint64_t) v10;
  TASSIGN(v60, v61);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v62;
  // pto: %cfirst
  uint64_t v63 = (uint64_t) v10;
  TASSIGN(v62, v63);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v64;
  // pto: %csecond
  uint64_t v65 = (uint64_t) v12;
  TASSIGN(v64, v65);
  TLOAD(v54, v24);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v56, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v58, v54);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v60, v56);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v62, v58, v60);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v48, v62);
  TMATMUL(v64, v58, v60);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v53, v64);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void different_row_stride(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c384
  const int64_t v8 = 384;
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
  int64_t v16 = v6 * v7;
  // pto: %tva
  int64_t v17 = v15 * v16;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v13, v14, v15, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v14 * v17, v17, v16, v7, v5);
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
  int64_t v28 = v7 * v7;
  // pto: %tvb
  int64_t v29 = v27 * v28;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v7, v5);
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
  int64_t v40 = v6 * v9;
  // pto: %tvo0
  int64_t v41 = v39 * v40;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v37, v38, v39, v6, v9);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v9, v5);
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
  int64_t v52 = v6 * v8;
  // pto: %tvo1
  int64_t v53 = v51 * v52;
  // pto: %tvo1
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v6, v9);
  // pto: %tvo1
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v8, v5);
  // pto: %tvo1
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v54, v55);
  // pto: %vo1
  __gm__ float* v57 = PTOAS__GLOBAL_TENSOR_DATA(v56);
  // pto: %vo1
  unsigned v58 = 128;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v59 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<24576, 24576, 24576, 384, 1> v60 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v57 + v58, v59, v60);
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

AICORE void non_power_two_stride(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c384
  const int64_t v8 = 384;
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
  int64_t v16 = v6 * v7;
  // pto: %tva
  int64_t v17 = v15 * v16;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v13, v14, v15, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v14 * v17, v17, v16, v7, v5);
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
  int64_t v28 = v7 * v7;
  // pto: %tvb
  int64_t v29 = v27 * v28;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v7, v5);
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
  int64_t v40 = v6 * v8;
  // pto: %tvo0
  int64_t v41 = v39 * v40;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v37, v38, v39, v6, v9);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v8, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v42, v43);
  // pto: %vo0
  __gm__ float* v45 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v46 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<24576, 24576, 24576, 384, 1> v47 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v45, v46, v47);
  // pto: %vo1
  __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo1
  unsigned v50 = 128;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v51 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<24576, 24576, 24576, 384, 1> v52 = pto::Stride<24576, 24576, 24576, 384, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v49 + v50, v51, v52);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v54;
  // pto: %am
  uint64_t v55 = (uint64_t) v10;
  TASSIGN(v54, v55);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v56;
  // pto: %bm
  uint64_t v57 = (uint64_t) v11;
  TASSIGN(v56, v57);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v58;
  // pto: %al
  uint64_t v59 = (uint64_t) v10;
  TASSIGN(v58, v59);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v60;
  // pto: %br
  uint64_t v61 = (uint64_t) v10;
  TASSIGN(v60, v61);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v62;
  // pto: %cfirst
  uint64_t v63 = (uint64_t) v10;
  TASSIGN(v62, v63);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v64;
  // pto: %csecond
  uint64_t v65 = (uint64_t) v12;
  TASSIGN(v64, v65);
  TLOAD(v54, v24);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v56, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v58, v54);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v60, v56);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v62, v58, v60);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v48, v62);
  TMATMUL(v64, v58, v60);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v53, v64);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
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

AICORE void unaligned_overlap(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %a16384
  const int64_t v10 = 16384;
  // pto: %a32768
  const int64_t v11 = 32768;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  int64_t v15 = v6 * v7;
  // pto: %tva
  int64_t v16 = v14 * v15;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v7, v5);
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
  int64_t v27 = v7 * v7;
  // pto: %tvb
  int64_t v28 = v26 * v27;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v7, v5);
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
  int64_t v39 = v6 * v8;
  // pto: %tvo0
  int64_t v40 = v38 * v39;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v6, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v8, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v41, v42);
  // pto: %vo0
  __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v45 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v46 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v44, v45, v46);
  // pto: %vo1
  __gm__ float* v48 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo1
  unsigned v49 = 4;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v50 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v51 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v52 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v48 + v49, v50, v51);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v53;
  // pto: %am
  uint64_t v54 = (uint64_t) v9;
  TASSIGN(v53, v54);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55;
  // pto: %bm
  uint64_t v56 = (uint64_t) v10;
  TASSIGN(v55, v56);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57;
  // pto: %al
  uint64_t v58 = (uint64_t) v9;
  TASSIGN(v57, v58);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v59;
  // pto: %br
  uint64_t v60 = (uint64_t) v9;
  TASSIGN(v59, v60);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v61;
  // pto: %cfirst
  uint64_t v62 = (uint64_t) v9;
  TASSIGN(v61, v62);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v63;
  // pto: %csecond
  uint64_t v64 = (uint64_t) v11;
  TASSIGN(v63, v64);
  TLOAD(v53, v23);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v55, v35);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v53);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v59, v55);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v61, v57, v59);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v61);
  TMATMUL(v63, v57, v59);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v52, v63);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void partial_column_overlap(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %a16384
  const int64_t v10 = 16384;
  // pto: %a32768
  const int64_t v11 = 32768;
  // pto: %tva
  const int64_t v12 = 1;
  // pto: %tva
  const int64_t v13 = 1;
  // pto: %tva
  const int64_t v14 = 1;
  // pto: %tva
  int64_t v15 = v6 * v7;
  // pto: %tva
  int64_t v16 = v14 * v15;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v7, v5);
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
  int64_t v27 = v7 * v7;
  // pto: %tvb
  int64_t v28 = v26 * v27;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v7, v5);
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
  int64_t v39 = v6 * v8;
  // pto: %tvo0
  int64_t v40 = v38 * v39;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v36, v37, v38, v6, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v8, v5);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v41, v42);
  // pto: %vo0
  __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v45 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 1> v46 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v44, v45, v46);
  // pto: %vo1
  __gm__ float* v48 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %vo1
  unsigned v49 = 64;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v50 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 1> v51 = pto::Stride<16384, 16384, 16384, 256, 1>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v52 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v48 + v49, v50, v51);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v53;
  // pto: %am
  uint64_t v54 = (uint64_t) v9;
  TASSIGN(v53, v54);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55;
  // pto: %bm
  uint64_t v56 = (uint64_t) v10;
  TASSIGN(v55, v56);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57;
  // pto: %al
  uint64_t v58 = (uint64_t) v9;
  TASSIGN(v57, v58);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v59;
  // pto: %br
  uint64_t v60 = (uint64_t) v9;
  TASSIGN(v59, v60);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v61;
  // pto: %cfirst
  uint64_t v62 = (uint64_t) v9;
  TASSIGN(v61, v62);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v63;
  // pto: %csecond
  uint64_t v64 = (uint64_t) v11;
  TASSIGN(v63, v64);
  TLOAD(v53, v23);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v55, v35);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v53);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v59, v55);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v61, v57, v59);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v47, v61);
  TMATMUL(v63, v57, v59);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v52, v63);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
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

AICORE void non_unit_inner_stride(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ float* v4) {
  using T = float;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c64
  const int64_t v6 = 64;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %c2
  const int64_t v9 = 2;
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
  int64_t v16 = v6 * v7;
  // pto: %tva
  int64_t v17 = v15 * v16;
  // pto: %tva
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v13, v14, v15, v6, v7);
  // pto: %tva
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v14 * v17, v17, v16, v7, v5);
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
  int64_t v28 = v7 * v7;
  // pto: %tvb
  int64_t v29 = v27 * v28;
  // pto: %tvb
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v7, v7);
  // pto: %tvb
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v7, v5);
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
  int64_t v40 = v6 * v8;
  // pto: %tvo0
  int64_t v41 = v39 * v40;
  // pto: %tvo0
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v37, v38, v39, v6, v8);
  // pto: %tvo0
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v8, v9);
  // pto: %tvo0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v42, v43);
  // pto: %vo0
  __gm__ float* v45 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo0
  pto::Shape<1, 1, 1, 64, 128> v46 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo0
  pto::Stride<16384, 16384, 16384, 256, 2> v47 = pto::Stride<16384, 16384, 16384, 256, 2>();
  // pto: %vo0
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 2>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 2>, pto::Layout::ND>(v45, v46, v47);
  // pto: %vo1
  __gm__ float* v49 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %vo1
  unsigned v50 = 256;
  // pto: %vo1
  pto::Shape<1, 1, 1, 64, 128> v51 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %vo1
  pto::Stride<16384, 16384, 16384, 256, 2> v52 = pto::Stride<16384, 16384, 16384, 256, 2>();
  // pto: %vo1
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 2>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<16384, 16384, 16384, 256, 2>, pto::Layout::ND>(v49 + v50, v51, v52);
  // pto: %am
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v54;
  // pto: %am
  uint64_t v55 = (uint64_t) v10;
  TASSIGN(v54, v55);
  // pto: %bm
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v56;
  // pto: %bm
  uint64_t v57 = (uint64_t) v11;
  TASSIGN(v56, v57);
  // pto: %al
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v58;
  // pto: %al
  uint64_t v59 = (uint64_t) v10;
  TASSIGN(v58, v59);
  // pto: %br
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v60;
  // pto: %br
  uint64_t v61 = (uint64_t) v10;
  TASSIGN(v60, v61);
  // pto: %cfirst
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v62;
  // pto: %cfirst
  uint64_t v63 = (uint64_t) v10;
  TASSIGN(v62, v63);
  // pto: %csecond
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v64;
  // pto: %csecond
  uint64_t v65 = (uint64_t) v12;
  TASSIGN(v64, v65);
  TLOAD(v54, v24);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v56, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v58, v54);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v60, v56);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v62, v58, v60);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v48, v62);
  TMATMUL(v64, v58, v60);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pipe_barrier(PIPE_FIX);
  TSTORE(v53, v64);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
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