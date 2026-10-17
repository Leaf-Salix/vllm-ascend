      TASSIGN(v3024, v3025);
      // pto: %stage_8_v31__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3026 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v438, v437);
      // pto: %stage_8_v31__ssa_v0
      ;
      uint64_t v3027 = (uint64_t) v316;
      TASSIGN(v3026, v3027);
      pipe_barrier(PIPE_V);
      TROWSUM(v3026, v3022, v3024);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %stage_8_v32__rv_v2
      ;
      float v3028;
      v3028 = v485;
      for (int64_t v3029 = v474; v3029 < v438; v3029 += v437) {
        // pto: %stage_8_v44__ssa_v0
        ;
        float v3030 = v3026.GetValue(v3029);
        // pto: %1014
        ;
        float v3031 = v3028 + v3030;
        v3028 = v3031;
      };
      // pto: %stage_8_v34__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3032 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v437, v438);
      // pto: %stage_8_v34__ssa_v0
      ;
      uint64_t v3033 = (uint64_t) v319;
      TASSIGN(v3032, v3033);
      pipe_barrier(PIPE_V);
      TEXPANDS(v3032, v485);
      // pto: %stage_8_v35__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3034 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v437, v438);
      // pto: %stage_8_v35__ssa_v0
      ;
      uint64_t v3035 = (uint64_t) v319;
      TASSIGN(v3034, v3035);
      pipe_barrier(PIPE_V);
      TADDS(v3034, v3032, v3028);
      // pto: %stage_8_v36__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3036 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v437, v438);
      // pto: %stage_8_v36__ssa_v0
      ;
      uint64_t v3037 = (uint64_t) v319;
      TASSIGN(v3036, v3037);
      pipe_barrier(PIPE_V);
      TMULS(v3036, v3034, v497);
      // pto: %stage_8_v37__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3038 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v437, v438);
      // pto: %stage_8_v37__ssa_v0
      ;
      uint64_t v3039 = (uint64_t) v319;
      TASSIGN(v3038, v3039);
      pipe_barrier(PIPE_V);
      TADDS(v3038, v3036, v487);
      // pto: %stage_8_v38__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3040 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v437, v438);
      // pto: %stage_8_v38__ssa_v0
      ;
      uint64_t v3041 = (uint64_t) v361;
      TASSIGN(v3040, v3041);
      // pto: %stage_8_v39__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v3042 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v437, v438);
      // pto: %stage_8_v39__ssa_v0
      ;
      uint64_t v3043 = (uint64_t) v362;
      TASSIGN(v3042, v3043);
      pipe_barrier(PIPE_V);
      TRSQRT(v3042, v3038, v3040);
      set_flag(PIPE_V, PIPE_S, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
      // pto: %stage_8_v40__ssa_v0
      ;
      float v3044 = v3042.GetValue(v474);
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      // pto: %stage_8_v41__ssa_v0
