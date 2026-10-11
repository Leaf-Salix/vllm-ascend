// Generated boundary excerpt; not a standalone translation unit.
  // pto: %6
  int64_t v202 = (int64_t) ((uint64_t) v201 * (uint64_t) v61);
  // pto: %7
  // pto: %leaf_multiple_inline11843_inline19368__phi_v6
  int64_t v203;
  if (v202 == v62) {
    v203 = v61;
  } else {
    // pto: %10
    ;
    // pto: %leaf_multiple_inline11843_inline19368__phi_v5
    ;
    int64_t v204;
    if (v202 == v63) {
      v204 = v61;
    } else {
      // pto: %12, %13, %leaf_multiple_inline11843_inline19368__phi_v4
      ;
      v204 = (int64_t) ((uint64_t) v201 * (uint64_t) v60) == v63 ? v60 : v50;
    };
    v203 = v204;
  }
  // pto: %14
  bool v205 = v203 > v50;
  // pto: %extra_leaves_inline11844_inline19249__phi_v3
  int64_t v206;
  // pto: %leaf_count_inline11847_inline19308__phi_v3
  int64_t v207;
  // pto: %leaf_tiles_inline11845_inline19266__phi_v3
  int64_t v208;
  if (v205) {
    // pto: %15
    ;
    uint64_t v209 = (uint64_t) v203;
    // pto: %15, %16, %17
    ;
    int64_t v210 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v200 + v209) - (uint64_t) v50) / v203;
    // pto: %18
    ;
    uint64_t v211 = (uint64_t) v203;
    // pto: %18
    ;
    int64_t v212 = (int64_t) ((uint64_t) v210 * v211);
    // pto: %19
    ;
    bool v213 = v212 <= v64;
    // pto: %extra_leaves_inline11844_inline19249__phi_v2
    ;
    // pto: %leaf_count_inline11847_inline19308__phi_v2
    ;
    int64_t v214;
    // pto: %leaf_tiles_inline11845_inline19266__phi_v2
    ;
    int64_t v215;
    if (v213) {
      // pto: %20
      ;
      int64_t v216 = v199 / v212;
      // pto: %21
      ;
      // pto: %22
      ;
      v214 = v199 % v212;
      v215 = v216 < v50 ? v50 : v216;
    } else {
      v214 = v65;
      v215 = v66;
    };
    v206 = v214;
    v207 = v213 ? v212 : v200;
    v208 = v215;
  } else {
    v206 = v65;
    v207 = v200;
    v208 = v66;
  }
  pypto_keep_prefix(v197);
  pypto_keep_prefix(v206);
  pypto_keep_prefix(v207);
  pypto_keep_prefix(v208);
  int64_t v217 = pypto_wait_local_boundary(v21, v22);
  const int64_t v218 = 0;
  if (v217 == v218) {
    return;
  }
  uint64_t v219 = reinterpret_cast<uint64_t>(v2);
  // pto: %24
  uint64_t v220 = (uint64_t) v207;
  // pto: %24
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int64_t v221 = v197; v221 < ((int64_t) ((uint64_t) v201 * v220)); v221 += v63) {
