// End of FileCheck-validated generated output.
  int64_t v14 = v7 * v9;
  // pto: %shape_src
  int64_t v15 = v13 * v14;
  // pto: %shape_src
  pto::Shape<1, 1, 1, 1, 32> v16 = pto::Shape<1, 1, 1, 1, 32>(v11, v12, v13, v7, v9);
  // pto: %shape_src
  pto::Stride<-1, -1, -1, -1, -1> v17 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v15, v15, v14, v9, v7);
  // pto: %shape_src
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v18 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v16, v17);
  // pto: %shape_dst
  const int64_t v19 = 1;
  // pto: %shape_dst
  const int64_t v20 = 1;
  // pto: %shape_dst
  const int64_t v21 = 1;
  // pto: %shape_dst
  int64_t v22 = v7 * v9;
  // pto: %shape_dst
  int64_t v23 = v21 * v22;
  // pto: %shape_dst
  pto::Shape<1, 1, 1, 1, 32> v24 = pto::Shape<1, 1, 1, 1, 32>(v19, v20, v21, v7, v9);
  // pto: %shape_dst
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v23, v23, v22, v9, v7);
  // pto: %shape_dst
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v24, v25);
  // pto: %view_src
  __gm__ float* v27 = PTOAS__GLOBAL_TENSOR_DATA(v18);
  // pto: %view_src
  pto::Shape<1, 1, 1, 1, 32> v28 = pto::Shape<1, 1, 1, 1, 32>();
  // pto: %view_src
  pto::Stride<32, 32, 32, 32, 1> v29 = pto::Stride<32, 32, 32, 32, 1>();
  // pto: %view_src
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v27, v28, v29);
  // pto: %view_dst
  __gm__ float* v31 = PTOAS__GLOBAL_TENSOR_DATA(v26);
  // pto: %view_dst
  pto::Shape<1, 1, 1, 1, 32> v32 = pto::Shape<1, 1, 1, 1, 32>();
  // pto: %view_dst
  pto::Stride<32, 32, 32, 32, 1> v33 = pto::Stride<32, 32, 32, 32, 1>();
  // pto: %view_dst
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v31, v32, v33);
  // pto: %tile
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
  // pto: %tile
  uint64_t v36 = (uint64_t) v10;
  TASSIGN(v35, v36);
  // pto: %positive, %limit
  // pto: %work
  int64_t v37;
  v37 = v7;
  for (int64_t v38 = v6; v38 < (v5 > v6 ? v5 : v7); v38 += v7) {
    // pto: %next
    ;
    uint64_t v39 = (uint64_t) v37;
    // pto: %next
    ;
    v37 = (int64_t) (v39 + (uint64_t) v8);
  }
  pypto_keep_prefix(v37);
  int64_t v40 = pypto_wait_local_boundary(v3, v4);
  const int64_t v41 = 0;
  if (v40 == v41) {
    return;
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v42 = v6; v42 < v37; v42 += v7) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v35, v30);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v34, v35);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  pypto_set_next_task_start(v3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kRuntimeBarrierAll);
  return;
}
