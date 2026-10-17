          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          // pto: %stage_16_v104__ssa_v0
          ;
          Tile<TileType::Acc, int32_t, 384, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v2680 = Tile<TileType::Acc, int32_t, 384, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v310, v372);
          // pto: %stage_16_v104__ssa_v0
          ;
          uint64_t v2681 = (uint64_t) v281;
          TASSIGN(v2680, v2681);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
          TMATMUL(v2680, v2404, v2598);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %stage_16_v105__ssa_v0
          ;
          Tile<TileType::Right, half, 384, 80, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v2682 = Tile<TileType::Right, half, 384, 80, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v310, v372);
          // pto: %stage_16_v105__ssa_v0
          ;
          uint64_t v2683 = (uint64_t) v281;
          TASSIGN(v2682, v2683);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          TMOV(v2682, v2591);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
          // pto: %stage_16_v106__ssa_v0
          ;
          Tile<TileType::Acc, float, 16, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v2684 = Tile<TileType::Acc, float, 16, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v315, v372);
          // pto: %stage_16_v106__ssa_v0
          ;
          uint64_t v2685 = (uint64_t) v302;
          TASSIGN(v2684, v2685);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          TMATMUL(v2684, v2416, v2682);
          set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          // pto: %stage_16_v107__ssa_v0
          ;
          Tile<TileType::Mat, half, 384, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v2686 = Tile<TileType::Mat, half, 384, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v310, v372);
          // pto: %stage_16_v107__ssa_v0
          ;
          uint64_t v2687 = (uint64_t) v283;
          TASSIGN(v2686, v2687);
          wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
          uint64_t v2688 = static_cast<uint64_t>(v374);
          uint16_t v2689 = static_cast<uint16_t>(v340);
          uint16_t v2690 = static_cast<uint16_t>(v340);
          TINSERT<Tile<TileType::Mat, half, 384, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Acc, int32_t, 384, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, ReluPreMode::NormalRelu>(v2686, v2680, v2688, v2689, v2690);
          set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
          v2684.SetValidShape(v341, v372);
          // pto: %1446
          ;
          int64_t v2691 = v2421 < v340 ? v340 : v2421;
          // pto: %temporary_49__ssa_v0_pview
          ;
          __gm__ float* v2692 = PTOAS__GLOBAL_TENSOR_DATA(v685);
          // pto: %temporary_49__ssa_v0_pview
          ;
          const int64_t v2693 = 0;

// Last WS
          TASSIGN(v3731, v3732);
          wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          TMOV(v3731, v3720);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %stage_16_v266__ssa_v0
          ;
          Tile<TileType::Acc, float, 16, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v3733 = Tile<TileType::Acc, float, 16, 80, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v315, v372);
          // pto: %stage_16_v266__ssa_v0
          ;
          uint64_t v3734 = (uint64_t) v281;
          TASSIGN(v3733, v3734);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          TMATMUL(v3733, v2416, v3731);
          set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          v3733.SetValidShape(v341, v318);
          // pto: %2135
          ;
          __gm__ float* v3735 = PTOAS__GLOBAL_TENSOR_DATA(v685);
          // pto: %2135
          ;
          const int64_t v3736 = 960;
          // pto: %2135
          ;
          const int64_t v3737 = 1024;
          // pto: %2135
          ;
          pto::Shape<1, 1, 1, 6, 64> v3738 = pto::Shape<1, 1, 1, 6, 64>();
          // pto: %2135
          ;
          pto::Stride<6144, 6144, 6144, 1024, 1> v3739 = pto::Stride<6144, 6144, 6144, 1024, 1>();
          // pto: %2135
          ;
