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
  kAsyncAivEvent7 = 2,
  kAsyncAicEvent7 = 3,
  kAsyncAivAllPipesEvent7 = 4,
  kAsyncAicAllPipesEvent7 = 5,
  kRuntimeBarrierAll = 6,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kRuntimeBarrierAll:
    // The dynamic caller executes the ordinary PIPE_ALL before publishing FIN
    // or before a same-core successor may touch pipeline resources.
    break;
  // Every event targets Scalar, so internal uses of the same pair/ID have
  // already been consumed before Scalar can issue these tail markers.
  // The caller must join all listed engines before reusing this core.
  case PTOAutoSyncTailMode::kAsyncAicAllPipesEvent7:
    set_flag(PIPE_M, PIPE_S, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAivAllPipesEvent7:
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAicEvent7:
    // The caller drains Cube and FIX independently. Join every upstream
    // matrix pipeline without making Scalar wait for device completion.
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    set_flag(PIPE_M, PIPE_S, EVENT_ID7);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    break;
  case PTOAutoSyncTailMode::kAsyncAivEvent7:
    // Join MTE2/V/MTE3 without a Scalar wait. The caller owns the final
    // MTE3->S event and must consume it before publishing completion.
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // Diagnostic completion marker: the caller measures Vector separately
    // from MTE3 so a trailing store cannot be mistaken for compute overlap.
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    break;
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

// PYPTO_TASK_BOUNDARY_JOINS_V1 _proj_b_mm_nz_kernel__3 mask=7
#ifndef PYPTO_TASK_BOUNDARY_JOINS_V1
#error reduced tail joins require a matching PyPTO wrapper
#endif
extern "C" AICORE void pypto_keep_prefix(int64_t);
extern "C" AICORE void pypto_wait_pre_task_end(__gm__ int64_t*, int64_t);
extern "C" AICORE void pypto_set_next_task_start(__gm__ int64_t*);
AICORE void _proj_b_mm_nz_kernel__3(__gm__ int8_t* v1, __gm__ int8_t* v2, __gm__ int32_t* v3, int64_t v4, int64_t v5, int64_t v6, __gm__ int64_t* v7, int64_t v8, int32_t v9, int32_t v10) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c64_i64
  const int64_t v11 = 64;
  // pto: %c262208_i64
  const int64_t v12 = 262208;
  // pto: %c0_i64
  const int64_t v13 = 0;
  // pto: %c16384_i64
  const int64_t v14 = 16384;
  // pto: %c32768_i64
  const int64_t v15 = 32768;
  // pto: %c384_index
  const int64_t v16 = 384;
  // pto: %c8192_index
  const int64_t v17 = 8192;
  // pto: %c1_index
  const int64_t v18 = 1;
  // pto: %c128_index
  const int64_t v19 = 128;
  // pto: %c512_index
  const int64_t v20 = 512;
  // pto: %c16_index
  const int64_t v21 = 16;
  // pto: %c32_index
  const int64_t v22 = 32;
  // pto: %c33554432_index
  const int64_t v23 = 33554432;
  // pto: %c262144_index
  const int64_t v24 = 262144;
  // pto: %c32768_index
  const int64_t v25 = 32768;
  // pto: %c0_index
  const int64_t v26 = 0;
  // pto: %c2_index
  const int64_t v27 = 2;
  // pto: %c256_index
  const int64_t v28 = 256;
  // pto: %c1024_index
  const int64_t v29 = 1024;
  // pto: %cn128_index
  const int64_t v30 = -128;
  // pto: %c4096_index
  const int64_t v31 = 4096;
  // pto: %o_r_i8_pad__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %o_r_i8_pad__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %o_r_i8_pad__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %o_r_i8_pad__ssa_v0_view
  int64_t v35 = v16 * v17;
  // pto: %o_r_i8_pad__ssa_v0_view
  int64_t v36 = v34 * v35;
  // pto: %o_r_i8_pad__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v16, v17);
  // pto: %o_r_i8_pad__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v36, v36, v35, v17, v18);
  // pto: %o_r_i8_pad__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v37, v38);
  // pto: %wo_b__ssa_v0_view
  pto::Shape<-1, -1, -1, -1, -1> v40 = pto::Shape<-1, -1, -1, -1, -1>(v18, v19, v20, v21, v22);
  // pto: %wo_b__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v23, v24, v20, v22, v18);
  // pto: %wo_b__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ> v42 = GlobalTensor<int8_t, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::NZ>(v2, v40, v41);
  // pto: %partials__ssa_v0_view
  const int64_t v43 = 1;
  // pto: %partials__ssa_v0_view
  const int64_t v44 = 1;
  // pto: %partials__ssa_v0_view
  const int64_t v45 = 1;
  // pto: %partials__ssa_v0_view
  int64_t v46 = v16 * v25;
  // pto: %partials__ssa_v0_view
  int64_t v47 = v45 * v46;
  // pto: %partials__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v48 = pto::Shape<1, 1, 1, -1, -1>(v43, v44, v45, v16, v25);
  // pto: %partials__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v49 = pto::Stride<-1, -1, -1, -1, -1>(v44 * v47, v47, v46, v25, v18);
  // pto: %partials__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v50 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v48, v49);
  // pto: %t__tmp_v0, %3
  int64_t v51 = (int64_t) ((uint64_t) ((int64_t) v9) * (uint64_t) v20);
  pypto_keep_prefix(v51);
  int64_t v52 = pypto_wait_local_boundary(v7, v8);
  const int64_t v53 = 0;
  if (v52 == v53) {
    return;
  }
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (int64_t v54 = v26; v54 < v27; v54 += v18) {
    // pto: %5, %4
    ;
    int64_t v55 = (int64_t) ((uint64_t) v51 + (uint64_t) ((int64_t) (uint64_t) v54 * (uint64_t) v28));
    // pto: %6
    ;
    int64_t v56 = v4 < v26 ? v26 : v4;
    // pto: %weight_resident__tile
    ;
    Tile<TileType::Mat, int8_t, 1024, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Mat, int8_t, 1024, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v29, v28);
    // pto: %weight_resident__tile
    ;
    uint64_t v58 = (uint64_t) v11;
    TASSIGN(v57, v58);
    // pto: %8
    ;
    int64_t v59 = v55 / v22;
    // pto: %9
    ;
    int64_t v60 = v59 < v26 ? v26 : v59;
    // pto: %7, %10
    ;
    int64_t v61 = (int64_t) ((uint64_t) v56 * (uint64_t) v29) / v21;
    // pto: %11
    ;
    int64_t v62 = v61 < v26 ? v26 : v61;
    // pto: %wo_b__ssa_v0_pview
    ;
    __gm__ int8_t* v63 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %wo_b__ssa_v0_pview
    ;
    const int64_t v64 = 0;
    // pto: %wo_b__ssa_v0_pview
    ;
    const int64_t v65 = 262144;
    // pto: %wo_b__ssa_v0_pview
    ;
    const int64_t v66 = 512;
    // pto: %wo_b__ssa_v0_pview
    ;
    pto::Shape<1, 8, 64, 16, 32> v67 = pto::Shape<1, 8, 64, 16, 32>();
    // pto: %wo_b__ssa_v0_pview
    ;
    pto::Stride<33554432, 262144, 512, 32, 1> v68 = pto::Stride<33554432, 262144, 512, 32, 1>();
    // pto: %wo_b__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 8, 64, 16, 32>, pto::Stride<33554432, 262144, 512, 32, 1>, pto::Layout::NZ> v69 = GlobalTensor<int8_t, pto::Shape<1, 8, 64, 16, 32>, pto::Stride<33554432, 262144, 512, 32, 1>, pto::Layout::NZ>(v63 + (v64 + v60 * v65 + v62 * v66), v67, v68);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v57, v69);
    for (int64_t v70 = v26; v70 < v6; v70 += v18) {
      // pto: %12
      ;
      int64_t v71 = (int64_t) ((uint64_t) v70 * (uint64_t) v19);
      // pto: %b_act_full__tile
      ;
      Tile<TileType::Mat, int8_t, 128, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Mat, int8_t, 128, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v29);
      // pto: %b_act_full__tile
      ;
      uint64_t v73 = (uint64_t) v12;
      TASSIGN(v72, v73);
      // pto: %13
      ;
      int64_t v74 = v71 < v26 ? v26 : v71;
      // pto: %14
      ;
      int64_t v75 = v5 < v26 ? v26 : v5;
      // pto: %o_r_i8_pad__ssa_v0_pview
      ;
      __gm__ int8_t* v76 = PTOAS__GLOBAL_TENSOR_DATA(v39);
      // pto: %o_r_i8_pad__ssa_v0_pview
      ;
      const int64_t v77 = 0;
      // pto: %o_r_i8_pad__ssa_v0_pview
      ;
      const int64_t v78 = 8192;
      // pto: %o_r_i8_pad__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 128, 1024> v79 = pto::Shape<1, 1, 1, 128, 1024>();
      // pto: %o_r_i8_pad__ssa_v0_pview
      ;
      pto::Stride<1048576, 1048576, 1048576, 8192, 1> v80 = pto::Stride<1048576, 1048576, 1048576, 8192, 1>();
      // pto: %o_r_i8_pad__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 1024>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND> v81 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 128, 1024>, pto::Stride<1048576, 1048576, 1048576, 8192, 1>, pto::Layout::ND>(v76 + (v77 + v74 * v78 + v75), v79, v80);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v72, v81);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %full_acc__tile_l0_init
      ;
      Tile<TileType::Acc, int32_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Acc, int32_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v19, v28);
      // pto: %full_acc__tile_l0_init
      ;
      uint64_t v83 = (uint64_t) v13;
      TASSIGN(v82, v83);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      for (int64_t v84 = v26; v84 < v29; v84 += v28) {
        // pto: %full_acc__tile_l0_a
        ;
        Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v19);
        // pto: %full_acc__tile_l0_a
        ;
        uint64_t v86 = (uint64_t) v13;
        TASSIGN(v85, v86);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        uint16_t v87 = static_cast<uint16_t>(v26);
        uint16_t v88 = static_cast<uint16_t>(v84);
        TEXTRACT(v85, v72, v87, v88);
        // pto: %full_acc__tile_l0_b
        ;
        Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v19, v28);
        // pto: %full_acc__tile_l0_b
        ;
        uint64_t v90 = (uint64_t) v13;
        TASSIGN(v89, v90);
        uint16_t v91 = static_cast<uint16_t>(v84);
        uint16_t v92 = static_cast<uint16_t>(v26);
        TEXTRACT(v89, v57, v91, v92);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %0
        ;
        Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Left, int8_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v19);
        // pto: %0
        ;
        uint64_t v94 = (uint64_t) v14;
        TASSIGN(v93, v94);
        // pto: %15
        ;
        int64_t v95 = (int64_t) ((uint64_t) v84 + (uint64_t) v19);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        uint16_t v96 = static_cast<uint16_t>(v26);
        uint16_t v97 = static_cast<uint16_t>(v95);
        TEXTRACT(v93, v72, v96, v97);
        // pto: %1
        ;
        Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v19, v28);
        // pto: %1
        ;
        uint64_t v99 = (uint64_t) v15;
        TASSIGN(v98, v99);
        uint16_t v100 = static_cast<uint16_t>(v95);
        uint16_t v101 = static_cast<uint16_t>(v26);
        TEXTRACT(v98, v57, v100, v101);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %17
        ;
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v84 == v26) {
          pipe_barrier(PIPE_M);
          TMATMUL(v82, v85, v89);
        } else {
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v82, v82, v85, v89);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %18
        ;
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if (v84 == v30) {
          pipe_barrier(PIPE_M);
          TMATMUL(v82, v93, v98);
        } else {
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v82, v82, v93, v98);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %20, %21
      ;
      int64_t v102 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v4 * (uint64_t) v31) + (uint64_t) v55);
      // pto: %22
      ;
      int64_t v103 = v102 < v26 ? v26 : v102;
      // pto: %partials__iter_v3_pview
      ;
      __gm__ int32_t* v104 = PTOAS__GLOBAL_TENSOR_DATA(v50);
      // pto: %partials__iter_v3_pview
      ;
      const int64_t v105 = 0;
      // pto: %partials__iter_v3_pview
      ;
      const int64_t v106 = 32768;
      // pto: %partials__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 128, 256> v107 = pto::Shape<1, 1, 1, 128, 256>();
      // pto: %partials__iter_v3_pview
      ;
      pto::Stride<4194304, 4194304, 4194304, 32768, 1> v108 = pto::Stride<4194304, 4194304, 4194304, 32768, 1>();
      // pto: %partials__iter_v3_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<4194304, 4194304, 4194304, 32768, 1>, pto::Layout::ND> v109 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<4194304, 4194304, 4194304, 32768, 1>, pto::Layout::ND>(v104 + (v105 + v74 * v106 + v103), v107, v108);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      TSTORE(v109, v82);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  pypto_set_next_task_start(v7);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}
