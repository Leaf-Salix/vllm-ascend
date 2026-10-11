// Generated boundary excerpt; not a standalone translation unit.
  // pto: %pair_arena_inline3214_inline4202__ssa_v0_view
  const int64_t v89 = 1;
  // pto: %pair_arena_inline3214_inline4202__ssa_v0_view
  int64_t v90 = v38 * v39;
  // pto: %pair_arena_inline3214_inline4202__ssa_v0_view
  int64_t v91 = v89 * v90;
  // pto: %pair_arena_inline3214_inline4202__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v92 = pto::Shape<1, 1, 1, -1, -1>(v87, v88, v89, v38, v39);
  // pto: %pair_arena_inline3214_inline4202__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v93 = pto::Stride<-1, -1, -1, -1, -1>(v88 * v91, v91, v90, v39, v37);
  // pto: %pair_arena_inline3214_inline4202__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v94 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v92, v93);
  // pto: %single_roots_inline8720__phi_v15
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v15
  uint64_t v96 = (uint64_t) v29;
  TASSIGN(v95, v96);
  // pto: %single_roots_inline8720__phi_v6
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v6
  uint64_t v98 = (uint64_t) v29;
  TASSIGN(v97, v98);
  // pto: %single_new_root_inline8695__phi_v4
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v39);
  // pto: %single_new_root_inline8695__phi_v4
  uint64_t v100 = (uint64_t) v35;
  TASSIGN(v99, v100);
  // pto: %single_new_root_inline8695__phi_v3
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v39);
  // pto: %single_new_root_inline8695__phi_v3
  uint64_t v102 = (uint64_t) v35;
  TASSIGN(v101, v102);
  // pto: %single_roots_inline8720__phi_v5
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v5
  uint64_t v104 = (uint64_t) v29;
  TASSIGN(v103, v104);
  // pto: %single_roots_inline8720__phi_v10
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v10
  uint64_t v106 = (uint64_t) v29;
  TASSIGN(v105, v106);
  // pto: %single_new_root_inline8695__phi_v9
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v39);
  // pto: %single_new_root_inline8695__phi_v9
  uint64_t v108 = (uint64_t) v35;
  TASSIGN(v107, v108);
  // pto: %single_new_root_inline8695__phi_v8
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v39);
  // pto: %single_new_root_inline8695__phi_v8
  uint64_t v110 = (uint64_t) v35;
  TASSIGN(v109, v110);
  // pto: %single_roots_inline8720__phi_v9
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v9
  uint64_t v112 = (uint64_t) v29;
  TASSIGN(v111, v112);
  // pto: %single_roots_inline8720__phi_v14
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v14
  uint64_t v114 = (uint64_t) v29;
  TASSIGN(v113, v114);
  // pto: %single_new_root_inline8695__phi_v14
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v39);
  // pto: %single_new_root_inline8695__phi_v14
  uint64_t v116 = (uint64_t) v35;
  TASSIGN(v115, v116);
  // pto: %single_new_root_inline8695__phi_v13
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v37, v39);
  // pto: %single_new_root_inline8695__phi_v13
  uint64_t v118 = (uint64_t) v35;
  TASSIGN(v117, v118);
  // pto: %single_roots_inline8720__phi_v13
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v39);
  // pto: %single_roots_inline8720__phi_v13
  uint64_t v120 = (uint64_t) v29;
  TASSIGN(v119, v120);
  // pto: %buf_worker_inline8842__ssa_v0
  int64_t v121 = (int64_t) v22;
  pypto_keep_prefix(v121);
  int64_t v122 = pypto_wait_local_boundary(v20, v21);
  const int64_t v123 = 0;
  if (v122 == v123) {
    return;
  }
  // pto: %0
  int64_t v124 = v13 / v42;
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %buf_max_cache_len_inline8747__rv_v2
  int64_t v125;
  v125 = v41;
  for (int64_t v126 = v41; v126 < v124; v126 += v37) {
    // pto: %t__tile
    ;
