// Generated C++ excerpt, source lines 370..440.
  // pto: %wq_b_scale__ssa_v0_view
  int64_t v106 = v102 * v105;
  // pto: %wq_b_scale__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v107 = pto::Shape<1, 1, 1, 1, -1>(v100, v101, v102, v103, v36);
  // pto: %wq_b_scale__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v108 = pto::Stride<-1, -1, -1, -1, -1>(v101 * v106, v106, v105, v104, v34);
  // pto: %wq_b_scale__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v109 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v107, v108);
  // pto: %q_flat_inline8594__ssa_v0_view
  const int64_t v110 = 1;
  // pto: %q_flat_inline8594__ssa_v0_view
  const int64_t v111 = 1;
  // pto: %q_flat_inline8594__ssa_v0_view
  const int64_t v112 = 1;
  // pto: %q_flat_inline8594__ssa_v0_view
  int64_t v113 = (int64_t) v13;
  // pto: %q_flat_inline8594__ssa_v0_view
  int64_t v114 = v113 * v36;
  // pto: %q_flat_inline8594__ssa_v0_view
  int64_t v115 = v112 * v114;
  // pto: %q_flat_inline8594__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v116 = pto::Shape<1, 1, 1, -1, -1>(v110, v111, v112, v113, v36);
  // pto: %q_flat_inline8594__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v117 = pto::Stride<-1, -1, -1, -1, -1>(v111 * v115, v115, v114, v36, v34);
  // pto: %q_flat_inline8594__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v118 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v116, v117);
  // pto: %dq_worker_inline8560__ssa_v0
  int64_t v119 = (int64_t) v16;
  // pto: %54, %55, %56, %57
  // pto: %dq_head_tile_inline8574__phi_v4
  int64_t v120;
  if (v8 >= v37 & v8 % v37 == v38) {
    v120 = v35;
  } else {
    // pto: %58, %59, %60, %61, %dq_head_tile_inline8574__phi_v3
    ;
    v120 = v8 >= v39 & v8 % v39 == v38 ? v40 : v41;
  }
  pypto_keep_prefix(v119);
  pypto_keep_prefix(v120);
  int64_t v121 = pypto_wait_local_boundary(v14, v15);
  const int64_t v122 = 0;
  if (v121 == v122) {
    return;
  }
  // pto: %64
  int64_t v123 = v35 / v120;
  // pto: %62, %63, %65
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v124 = v119; v124 < ((int64_t) ((uint64_t) ((int64_t) ((uint64_t) v8 + (uint64_t) v42) / v43) * (uint64_t) v123)); v124 += v44) {
    // pto: %68
    ;
