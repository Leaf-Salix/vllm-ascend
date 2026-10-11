// Generated boundary excerpt; not a standalone translation unit.
  // pto: %pair_arena_inline3212_inline4208__ssa_v0_view
  const int64_t v90 = 1;
  // pto: %pair_arena_inline3212_inline4208__ssa_v0_view
  int64_t v91 = v39 * v40;
  // pto: %pair_arena_inline3212_inline4208__ssa_v0_view
  int64_t v92 = v90 * v91;
  // pto: %pair_arena_inline3212_inline4208__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v93 = pto::Shape<1, 1, 1, -1, -1>(v88, v89, v90, v39, v40);
  // pto: %pair_arena_inline3212_inline4208__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v94 = pto::Stride<-1, -1, -1, -1, -1>(v89 * v92, v92, v91, v40, v38);
  // pto: %pair_arena_inline3212_inline4208__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v95 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v93, v94);
  // pto: %single_roots_inline8871__phi_v15
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v15
  uint64_t v97 = (uint64_t) v30;
  TASSIGN(v96, v97);
  // pto: %single_roots_inline8871__phi_v6
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v6
  uint64_t v99 = (uint64_t) v30;
  TASSIGN(v98, v99);
  // pto: %single_new_root_inline8868__phi_v4
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %single_new_root_inline8868__phi_v4
  uint64_t v101 = (uint64_t) v36;
  TASSIGN(v100, v101);
  // pto: %single_new_root_inline8868__phi_v3
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %single_new_root_inline8868__phi_v3
  uint64_t v103 = (uint64_t) v36;
  TASSIGN(v102, v103);
  // pto: %single_roots_inline8871__phi_v5
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v5
  uint64_t v105 = (uint64_t) v30;
  TASSIGN(v104, v105);
  // pto: %single_roots_inline8871__phi_v10
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v10
  uint64_t v107 = (uint64_t) v30;
  TASSIGN(v106, v107);
  // pto: %single_new_root_inline8868__phi_v9
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %single_new_root_inline8868__phi_v9
  uint64_t v109 = (uint64_t) v36;
  TASSIGN(v108, v109);
  // pto: %single_new_root_inline8868__phi_v8
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %single_new_root_inline8868__phi_v8
  uint64_t v111 = (uint64_t) v36;
  TASSIGN(v110, v111);
  // pto: %single_roots_inline8871__phi_v9
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v9
  uint64_t v113 = (uint64_t) v30;
  TASSIGN(v112, v113);
  // pto: %single_roots_inline8871__phi_v14
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v14
  uint64_t v115 = (uint64_t) v30;
  TASSIGN(v114, v115);
  // pto: %single_new_root_inline8868__phi_v14
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %single_new_root_inline8868__phi_v14
  uint64_t v117 = (uint64_t) v36;
  TASSIGN(v116, v117);
  // pto: %single_new_root_inline8868__phi_v13
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %single_new_root_inline8868__phi_v13
  uint64_t v119 = (uint64_t) v36;
  TASSIGN(v118, v119);
  // pto: %single_roots_inline8871__phi_v13
  Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 3, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v47, v40);
  // pto: %single_roots_inline8871__phi_v13
  uint64_t v121 = (uint64_t) v30;
  TASSIGN(v120, v121);
  // pto: %buf_worker_inline8774__ssa_v0
  int64_t v122 = (int64_t) v23;
  pypto_keep_prefix(v122);
  int64_t v123 = pypto_wait_local_boundary(v21, v22);
  const int64_t v124 = 0;
  if (v123 == v124) {
    return;
  }
  // pto: %0
  int64_t v125 = v12 < v42 ? v12 : v42;
  // pto: %1, %2
  int64_t v126 = (int64_t) ((uint64_t) v125 + (uint64_t) v43) / v40;
  // pto: %3, %4
  int64_t v127 = (int64_t) ((uint64_t) v125 + (uint64_t) v44) / v45;
  // pto: %5
  int64_t v128 = v14 / v46;
  // pto: %6
  int64_t v129 = (int64_t) ((uint64_t) v128 * (uint64_t) v47);
