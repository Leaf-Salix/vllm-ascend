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

AICORE void boxed_disjoint(__gm__ int8_t* v1) {
  using T = float;
  const int64_t v2 = 2048;
  const int64_t v3 = 1024;
  // pto: %c0
  const int64_t v4 = 0;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c32
  const int64_t v6 = 32;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %a0
  const int64_t v8 = 0;
  // pto: %tv
  const int64_t v9 = 1;
  // pto: %tv
  const int64_t v10 = 1;
  // pto: %tv
  const int64_t v11 = 1;
  // pto: %tv
  int64_t v12 = v6 * v7;
  // pto: %tv
  int64_t v13 = v11 * v12;
  // pto: %tv
  pto::Shape<1, 1, 1, -1, -1> v14 = pto::Shape<1, 1, 1, -1, -1>(v9, v10, v11, v6, v7);
  // pto: %tv
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v7, v5);
  // pto: %tv
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %input
  __gm__ int8_t* v17 = PTOAS__GLOBAL_TENSOR_DATA(v16);
  // pto: %input
  pto::Shape<1, 1, 1, 32, 128> v18 = pto::Shape<1, 1, 1, 32, 128>();
  // pto: %input
  pto::Stride<4096, 4096, 4096, 128, 1> v19 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %input
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v20 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v17, v18, v19);
  // pto: %pool
  uint64_t v21 = (uint64_t) v8;
  // pto: %transposed
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v22;
  // pto: %transposed
  uint64_t v23 = (uint64_t) v8;
  TASSIGN(v22, v23);
  // pto: %right
  Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v24;
  // pto: %right
  uint64_t v25 = (uint64_t) v8;
  TASSIGN(v24, v25);
  // pto: %write_pool
  uint64_t v26 = (uint64_t) v8;
  // pto: %first
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v27;
  // pto: %first
  uint64_t v28 = (uint64_t) v8;
  TASSIGN(v27, v28);
  // pto: %second
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v29;
  // pto: %second
  uint64_t v30 = (uint64_t) v3;
  TASSIGN(v29, v30);
  // pto: %later
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v31;
  // pto: %later
  uint64_t v32 = (uint64_t) v2;
  TASSIGN(v31, v32);
  TLOAD(v27, v20);
  TLOAD(v29, v20);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  uint16_t v33 = static_cast<uint16_t>(v4);
  uint16_t v34 = static_cast<uint16_t>(v4);
  TEXTRACT(v24, v22, v33, v34);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  TLOAD(v31, v20);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void boxed_overlap(__gm__ int8_t* v1) {
  using T = float;
  const int64_t v2 = 1024;
  // pto: %c0
  const int64_t v3 = 0;
  // pto: %c1
  const int64_t v4 = 1;
  // pto: %c32
  const int64_t v5 = 32;
  // pto: %c128
  const int64_t v6 = 128;
  // pto: %a0
  const int64_t v7 = 0;
  // pto: %tv
  const int64_t v8 = 1;
  // pto: %tv
  const int64_t v9 = 1;
  // pto: %tv
  const int64_t v10 = 1;
  // pto: %tv
  int64_t v11 = v5 * v6;
  // pto: %tv
  int64_t v12 = v10 * v11;
  // pto: %tv
  pto::Shape<1, 1, 1, -1, -1> v13 = pto::Shape<1, 1, 1, -1, -1>(v8, v9, v10, v5, v6);
  // pto: %tv
  pto::Stride<-1, -1, -1, -1, -1> v14 = pto::Stride<-1, -1, -1, -1, -1>(v9 * v12, v12, v11, v6, v4);
  // pto: %tv
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v15 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v13, v14);
  // pto: %input
  __gm__ int8_t* v16 = PTOAS__GLOBAL_TENSOR_DATA(v15);
  // pto: %input
  pto::Shape<1, 1, 1, 32, 128> v17 = pto::Shape<1, 1, 1, 32, 128>();
  // pto: %input
  pto::Stride<4096, 4096, 4096, 128, 1> v18 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %input
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v19 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v16, v17, v18);
  // pto: %pool
  uint64_t v20 = (uint64_t) v7;
  // pto: %transposed
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v21;
  // pto: %transposed
  uint64_t v22 = (uint64_t) v7;
  TASSIGN(v21, v22);
  // pto: %right
  Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v23;
  // pto: %right
  uint64_t v24 = (uint64_t) v7;
  TASSIGN(v23, v24);
  // pto: %write_pool
  uint64_t v25 = (uint64_t) v7;
  // pto: %first
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26;
  // pto: %first
  uint64_t v27 = (uint64_t) v7;
  TASSIGN(v26, v27);
  // pto: %second
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v28;
  // pto: %second
  uint64_t v29 = (uint64_t) v2;
  TASSIGN(v28, v29);
  // pto: %later
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v30;
  // pto: %later
  uint64_t v31 = (uint64_t) v2;
  TASSIGN(v30, v31);
  TLOAD(v26, v19);
  TLOAD(v28, v19);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  uint16_t v32 = static_cast<uint16_t>(v3);
  uint16_t v33 = static_cast<uint16_t>(v3);
  TEXTRACT(v23, v21, v32, v33);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  TLOAD(v30, v19);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void separate_disjoint(__gm__ int8_t* v1) {
  using T = float;
  const int64_t v2 = 18432;
  const int64_t v3 = 1024;
  // pto: %c0
  const int64_t v4 = 0;
  // pto: %c1
  const int64_t v5 = 1;
  // pto: %c32
  const int64_t v6 = 32;
  // pto: %c128
  const int64_t v7 = 128;
  // pto: %a0
  const int64_t v8 = 0;
  // pto: %a16384
  const int64_t v9 = 16384;
  // pto: %tv
  const int64_t v10 = 1;
  // pto: %tv
  const int64_t v11 = 1;
  // pto: %tv
  const int64_t v12 = 1;
  // pto: %tv
  int64_t v13 = v6 * v7;
  // pto: %tv
  int64_t v14 = v12 * v13;
  // pto: %tv
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v10, v11, v12, v6, v7);
  // pto: %tv
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v14, v14, v13, v7, v5);
  // pto: %tv
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v17 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v15, v16);
  // pto: %input
  __gm__ int8_t* v18 = PTOAS__GLOBAL_TENSOR_DATA(v17);
  // pto: %input
  pto::Shape<1, 1, 1, 32, 128> v19 = pto::Shape<1, 1, 1, 32, 128>();
  // pto: %input
  pto::Stride<4096, 4096, 4096, 128, 1> v20 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %input
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v21 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v18, v19, v20);
  // pto: %pool
  uint64_t v22 = (uint64_t) v8;
  // pto: %transposed
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v23;
  // pto: %transposed
  uint64_t v24 = (uint64_t) v8;
  TASSIGN(v23, v24);
  // pto: %right
  Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v25;
  // pto: %right
  uint64_t v26 = (uint64_t) v8;
  TASSIGN(v25, v26);
  // pto: %write_pool
  uint64_t v27 = (uint64_t) v9;
  // pto: %first
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v28;
  // pto: %first
  uint64_t v29 = (uint64_t) v8;
  TASSIGN(v28, v29);
  // pto: %second
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v30;
  // pto: %second
  uint64_t v31 = (uint64_t) v3;
  TASSIGN(v30, v31);
  // pto: %later
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v32;
  // pto: %later
  uint64_t v33 = (uint64_t) v2;
  TASSIGN(v32, v33);
  TLOAD(v28, v21);
  TLOAD(v30, v21);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  uint16_t v34 = static_cast<uint16_t>(v4);
  uint16_t v35 = static_cast<uint16_t>(v4);
  TEXTRACT(v25, v23, v34, v35);
  TLOAD(v32, v21);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void dynamic_unknown(__gm__ int8_t* v1, int64_t v2) {
  using T = float;
  const int64_t v3 = 1024;
  const int64_t v4 = 32;
  // pto: %c0
  const int64_t v5 = 0;
  // pto: %c1
  const int64_t v6 = 1;
  // pto: %c32
  const int64_t v7 = 32;
  // pto: %c128
  const int64_t v8 = 128;
  // pto: %a0
  const int64_t v9 = 0;
  // pto: %tv
  const int64_t v10 = 1;
  // pto: %tv
  const int64_t v11 = 1;
  // pto: %tv
  const int64_t v12 = 1;
  // pto: %tv
  int64_t v13 = v7 * v8;
  // pto: %tv
  int64_t v14 = v12 * v13;
  // pto: %tv
  pto::Shape<1, 1, 1, -1, -1> v15 = pto::Shape<1, 1, 1, -1, -1>(v10, v11, v12, v7, v8);
  // pto: %tv
  pto::Stride<-1, -1, -1, -1, -1> v16 = pto::Stride<-1, -1, -1, -1, -1>(v11 * v14, v14, v13, v8, v6);
  // pto: %tv
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v17 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v15, v16);
  // pto: %input
  __gm__ int8_t* v18 = PTOAS__GLOBAL_TENSOR_DATA(v17);
  // pto: %input
  pto::Shape<1, 1, 1, 32, 128> v19 = pto::Shape<1, 1, 1, 32, 128>();
  // pto: %input
  pto::Stride<4096, 4096, 4096, 128, 1> v20 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %input
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v21 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v18, v19, v20);
  // pto: %pool
  uint64_t v22 = (uint64_t) v9;
  // pto: %transposed
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v23;
  // pto: %transposed
  uint64_t v24 = (uint64_t) v9;
  TASSIGN(v23, v24);
  // pto: %right
  Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v25;
  // pto: %right
  uint64_t v26 = (uint64_t) v9;
  TASSIGN(v25, v26);
  // pto: %write_pool
  uint64_t v27 = (uint64_t) v9;
  // pto: %first
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v28;
  // pto: %first
  uint64_t v29 = (uint64_t) v9;
  TASSIGN(v28, v29);
  // pto: %second
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v30;
  // pto: %second
  uint64_t v31 = (uint64_t) v3;
  TASSIGN(v30, v31);
  // pto: %later
  Tile<TileType::Mat, int8_t, 128, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v32;
  // pto: %later
  uint64_t v33 = (uint64_t) ((int64_t) (uint64_t) v2 * (uint64_t) v4);
  TASSIGN(v32, v33);
  TLOAD(v28, v21);
  TLOAD(v30, v21);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  uint16_t v34 = static_cast<uint16_t>(v5);
  uint16_t v35 = static_cast<uint16_t>(v5);
  TEXTRACT(v25, v23, v34, v35);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  TLOAD(v32, v21);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}