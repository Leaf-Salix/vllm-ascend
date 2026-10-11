// Generated boundary excerpt; not a standalone translation unit.
  // pto: %6
  int64_t v127 = (int64_t) ((uint64_t) v126 * (uint64_t) v47);
  // pto: %7
  // pto: %leaf_multiple_inline11843_inline19368__phi_v6
  int64_t v128;
  if (v127 == v48) {
    v128 = v47;
  } else {
    // pto: %10
    ;
    // pto: %leaf_multiple_inline11843_inline19368__phi_v5
    ;
    int64_t v129;
    if (v127 == v49) {
      v129 = v47;
    } else {
      // pto: %12, %13, %leaf_multiple_inline11843_inline19368__phi_v4
      ;
      v129 = (int64_t) ((uint64_t) v126 * (uint64_t) v46) == v49 ? v46 : v38;
    };
    v128 = v129;
  }
  // pto: %14
  bool v130 = v128 > v38;
  // pto: %extra_leaves_inline11844_inline19249__phi_v3
  int64_t v131;
  // pto: %leaf_count_inline11847_inline19308__phi_v3
  int64_t v132;
  // pto: %leaf_tiles_inline11845_inline19266__phi_v3
  int64_t v133;
  if (v130) {
    // pto: %15
    ;
    uint64_t v134 = (uint64_t) v128;
    // pto: %15, %16, %17
    ;
    int64_t v135 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v125 + v134) - (uint64_t) v38) / v128;
    // pto: %18
    ;
    uint64_t v136 = (uint64_t) v128;
    // pto: %18
    ;
    int64_t v137 = (int64_t) ((uint64_t) v135 * v136);
    // pto: %19
    ;
    bool v138 = v137 <= v50;
    // pto: %extra_leaves_inline11844_inline19249__phi_v2
    ;
    // pto: %leaf_count_inline11847_inline19308__phi_v2
    ;
    int64_t v139;
    // pto: %leaf_tiles_inline11845_inline19266__phi_v2
    ;
    int64_t v140;
    if (v138) {
      // pto: %20
      ;
      int64_t v141 = v124 / v137;
      // pto: %21
      ;
      // pto: %22
      ;
      v139 = v124 % v137;
      v140 = v141 < v38 ? v38 : v141;
    } else {
      v139 = v51;
      v140 = v52;
    };
    v131 = v139;
    v132 = v138 ? v137 : v125;
    v133 = v140;
  } else {
    v131 = v51;
    v132 = v125;
    v133 = v52;
  }
  pypto_keep_prefix(v122);
  pypto_keep_prefix(v131);
  pypto_keep_prefix(v132);
  pypto_keep_prefix(v133);
  int64_t v142 = pypto_wait_local_boundary(v21, v22);
  const int64_t v143 = 0;
  if (v142 == v143) {
    return;
  }
  uint64_t v144 = reinterpret_cast<uint64_t>(v2);
  // pto: %buf_score_lane_inline19195__ssa_v0
  int64_t v145 = (int64_t) v25;
  // pto: %24
  uint64_t v146 = (uint64_t) v132;
  // pto: %24
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v147 = v122; v147 < ((int64_t) ((uint64_t) v126 * v146)); v147 += v49) {
    // pto: %25
