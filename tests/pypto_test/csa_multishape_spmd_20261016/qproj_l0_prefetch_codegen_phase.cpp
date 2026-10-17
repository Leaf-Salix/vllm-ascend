// pto: %stage_5_v5__ssa_v0
    ;
    Tile<TileType::Mat, int8_t, 96, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1519 = Tile<TileType::Mat, int8_t, 96, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v328);
    // pto: %stage_5_v5__ssa_v0
    ;
    uint64_t v1520 = (uint64_t) v294;
    TASSIGN(v1519, v1520);
    // pto: %temporary_22__ssa_v0_pview
    ;
    __gm__ int8_t* v1521 = PTOAS__GLOBAL_TENSOR_DATA(v467);
    // pto: %temporary_22__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 96, 1024> v1522 = pto::Shape<1, 1, 1, 96, 1024>();
    // pto: %temporary_22__ssa_v0_pview
    ;
    pto::Stride<98304, 98304, 98304, 1024, 1> v1523 = pto::Stride<98304, 98304, 98304, 1024, 1>();
    // pto: %temporary_22__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 96, 1024>, pto::Stride<98304, 98304, 98304, 1024, 1>, pto::Layout::ND> v1524 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 96, 1024>, pto::Stride<98304, 98304, 98304, 1024, 1>, pto::Layout::ND>(v1521, v1522, v1523);
    TLOAD(v1519, v1524);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %660
    ;
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v1525 = v350; v1525 < ((int64_t) ((uint64_t) v270 - (uint64_t) v1501) / v361); v1525 += v317) {
      // pto: %662, %661, %663
      ;
      int64_t v1526 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v1501 + (uint64_t) ((int64_t) (uint64_t) v1525 * (uint64_t) v361)) * (uint64_t) v329);
      // pto: %stage_5_v8__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1527 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v8__ssa_v0
      ;
      uint64_t v1528 = (uint64_t) v292;
      TASSIGN(v1527, v1528);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      uint16_t v1529 = static_cast<uint16_t>(v350);
      uint16_t v1530 = static_cast<uint16_t>(v350);
      TEXTRACT(v1527, v1519, v1529, v1530);
      // pto: %stage_5_v9__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1531 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v9__ssa_v0
      ;
      uint64_t v1532 = (uint64_t) v303;
      TASSIGN(v1531, v1532);
      // pto: %664
      ;
      int64_t v1533 = v1526 / v321;
      // pto: %665
      ;
      int64_t v1534 = v1533 < v350 ? v350 : v1533;
      // pto: %arg_wq_b__ssa_v0_pview
      ;
      __gm__ int8_t* v1535 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %arg_wq_b__ssa_v0_pview
      ;
      const int64_t v1536 = 0;
      // pto: %arg_wq_b__ssa_v0_pview
      ;
      const int64_t v1537 = 32768;
      // pto: %arg_wq_b__ssa_v0_pview
      ;
      pto::Shape<1, 8, 8, 16, 32> v1538 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %arg_wq_b__ssa_v0_pview
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1539 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %arg_wq_b__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1540 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1535 + (v1536 + v1534 * v1537), v1538, v1539);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v1531, v1540);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v10__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1541 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v10__ssa_v0
      ;
      uint64_t v1542 = (uint64_t) v292;
      TASSIGN(v1541, v1542);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1541, v1531);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %stage_5_v11__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1543 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v11__ssa_v0
      ;
      uint64_t v1544 = (uint64_t) v307;
      TASSIGN(v1543, v1544);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      uint16_t v1545 = static_cast<uint16_t>(v350);
      uint16_t v1546 = static_cast<uint16_t>(v332);
      TEXTRACT(v1543, v1519, v1545, v1546);
      // pto: %stage_5_v12__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1547 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v12__ssa_v0
      ;
      uint64_t v1548 = (uint64_t) v303;
      TASSIGN(v1547, v1548);
      // pto: %668
      ;
      __gm__ int8_t* v1549 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %668
      ;
      const int64_t v1550 = 4096;
      // pto: %668
      ;
      const int64_t v1551 = 32768;
      // pto: %668
      ;
      pto::Shape<1, 8, 8, 16, 32> v1552 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %668
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1553 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %668
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1554 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1549 + (v1550 + v1534 * v1551), v1552, v1553);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1547, v1554);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v13__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1555 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v13__ssa_v0
      ;
      uint64_t v1556 = (uint64_t) v301;
      TASSIGN(v1555, v1556);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1555, v1547);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %stage_5_v14__ssa_v0
      ;
      Tile<TileType::Acc, int32_t, 96, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v1557 = Tile<TileType::Acc, int32_t, 96, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v316, v329);
      // pto: %stage_5_v14__ssa_v0
      ;
      uint64_t v1558 = (uint64_t) v292;
      TASSIGN(v1557, v1558);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      TMATMUL(v1557, v1527, v1541);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %stage_5_v15__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1559 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v15__ssa_v0
      ;
      uint64_t v1560 = (uint64_t) v292;
      TASSIGN(v1559, v1560);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v1561 = static_cast<uint16_t>(v350);
      uint16_t v1562 = static_cast<uint16_t>(v329);
      TEXTRACT(v1559, v1519, v1561, v1562);
      // pto: %stage_5_v16__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1563 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v16__ssa_v0
      ;
      uint64_t v1564 = (uint64_t) v303;
      TASSIGN(v1563, v1564);
      // pto: %671
      ;
      __gm__ int8_t* v1565 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %671
      ;
      const int64_t v1566 = 8192;
      // pto: %671
      ;
      const int64_t v1567 = 32768;
      // pto: %671
      ;
      pto::Shape<1, 8, 8, 16, 32> v1568 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %671
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1569 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %671
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1570 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1565 + (v1566 + v1534 * v1567), v1568, v1569);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1563, v1570);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v17__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1571 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v17__ssa_v0
      ;
      uint64_t v1572 = (uint64_t) v292;
      TASSIGN(v1571, v1572);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1571, v1563);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v1557, v1557, v1543, v1555);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %stage_5_v18__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1573 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v18__ssa_v0
      ;
      uint64_t v1574 = (uint64_t) v307;
      TASSIGN(v1573, v1574);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v1575 = static_cast<uint16_t>(v350);
      uint16_t v1576 = static_cast<uint16_t>(v320);
      TEXTRACT(v1573, v1519, v1575, v1576);
      // pto: %stage_5_v19__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1577 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v19__ssa_v0
      ;
      uint64_t v1578 = (uint64_t) v303;
      TASSIGN(v1577, v1578);
      // pto: %674
      ;
      __gm__ int8_t* v1579 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %674
      ;
      const int64_t v1580 = 12288;
      // pto: %674
      ;
      const int64_t v1581 = 32768;
      // pto: %674
      ;
      pto::Shape<1, 8, 8, 16, 32> v1582 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %674
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1583 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %674
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1584 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1579 + (v1580 + v1534 * v1581), v1582, v1583);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1577, v1584);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v20__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1585 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v20__ssa_v0
      ;
      uint64_t v1586 = (uint64_t) v301;
      TASSIGN(v1585, v1586);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1585, v1577);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL_ACC(v1557, v1557, v1559, v1571);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %stage_5_v21__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1587 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v21__ssa_v0
      ;
      uint64_t v1588 = (uint64_t) v292;
      TASSIGN(v1587, v1588);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v1589 = static_cast<uint16_t>(v350);
      uint16_t v1590 = static_cast<uint16_t>(v331);
      TEXTRACT(v1587, v1519, v1589, v1590);
      // pto: %stage_5_v22__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1591 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v22__ssa_v0
      ;
      uint64_t v1592 = (uint64_t) v303;
      TASSIGN(v1591, v1592);
      // pto: %677
      ;
      __gm__ int8_t* v1593 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %677
      ;
      const int64_t v1594 = 16384;
      // pto: %677
      ;
      const int64_t v1595 = 32768;
      // pto: %677
      ;
      pto::Shape<1, 8, 8, 16, 32> v1596 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %677
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1597 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %677
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1598 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1593 + (v1594 + v1534 * v1595), v1596, v1597);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1591, v1598);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v23__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1599 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v23__ssa_v0
      ;
      uint64_t v1600 = (uint64_t) v292;
      TASSIGN(v1599, v1600);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1599, v1591);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v1557, v1557, v1573, v1585);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %stage_5_v24__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1601 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v24__ssa_v0
      ;
      uint64_t v1602 = (uint64_t) v307;
      TASSIGN(v1601, v1602);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v1603 = static_cast<uint16_t>(v350);
      uint16_t v1604 = static_cast<uint16_t>(v344);
      TEXTRACT(v1601, v1519, v1603, v1604);
      // pto: %stage_5_v25__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1605 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v25__ssa_v0
      ;
      uint64_t v1606 = (uint64_t) v303;
      TASSIGN(v1605, v1606);
      // pto: %680
      ;
      __gm__ int8_t* v1607 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %680
      ;
      const int64_t v1608 = 20480;
      // pto: %680
      ;
      const int64_t v1609 = 32768;
      // pto: %680
      ;
      pto::Shape<1, 8, 8, 16, 32> v1610 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %680
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1611 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %680
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1612 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1607 + (v1608 + v1534 * v1609), v1610, v1611);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1605, v1612);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v26__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1613 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v26__ssa_v0
      ;
      uint64_t v1614 = (uint64_t) v301;
      TASSIGN(v1613, v1614);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1613, v1605);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL_ACC(v1557, v1557, v1587, v1599);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %stage_5_v27__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1615 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v27__ssa_v0
      ;
      uint64_t v1616 = (uint64_t) v292;
      TASSIGN(v1615, v1616);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v1617 = static_cast<uint16_t>(v350);
      uint16_t v1618 = static_cast<uint16_t>(v362);
      TEXTRACT(v1615, v1519, v1617, v1618);
      // pto: %stage_5_v28__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1619 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v28__ssa_v0
      ;
      uint64_t v1620 = (uint64_t) v303;
      TASSIGN(v1619, v1620);
      // pto: %683
      ;
      __gm__ int8_t* v1621 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %683
      ;
      const int64_t v1622 = 24576;
      // pto: %683
      ;
      const int64_t v1623 = 32768;
      // pto: %683
      ;
      pto::Shape<1, 8, 8, 16, 32> v1624 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %683
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1625 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %683
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1626 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1621 + (v1622 + v1534 * v1623), v1624, v1625);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1619, v1626);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v29__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1627 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v29__ssa_v0
      ;
      uint64_t v1628 = (uint64_t) v292;
      TASSIGN(v1627, v1628);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1627, v1619);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v1557, v1557, v1601, v1613);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %stage_5_v30__ssa_v0
      ;
      Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1629 = Tile<TileType::Left, int8_t, 96, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v316, v332);
      // pto: %stage_5_v30__ssa_v0
      ;
      uint64_t v1630 = (uint64_t) v307;
      TASSIGN(v1629, v1630);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      uint16_t v1631 = static_cast<uint16_t>(v350);
      uint16_t v1632 = static_cast<uint16_t>(v363);
      TEXTRACT(v1629, v1519, v1631, v1632);
      // pto: %stage_5_v31__ssa_v0
      ;
      Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v1633 = Tile<TileType::Mat, int8_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v31__ssa_v0
      ;
      uint64_t v1634 = (uint64_t) v303;
      TASSIGN(v1633, v1634);
      // pto: %686
      ;
      __gm__ int8_t* v1635 = PTOAS__GLOBAL_TENSOR_DATA(v486);
      // pto: %686
      ;
      const int64_t v1636 = 28672;
      // pto: %686
      ;
      const int64_t v1637 = 32768;
      // pto: %686
      ;
      pto::Shape<1, 8, 8, 16, 32> v1638 = pto::Shape<1, 8, 8, 16, 32>();
      // pto: %686
      ;
      pto::Stride<33554432, 32768, 512, 32, 1> v1639 = pto::Stride<33554432, 32768, 512, 32, 1>();
      // pto: %686
      ;
      GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ> v1640 = GlobalTensor<int8_t, pto::Shape<1, 8, 8, 16, 32>, pto::Stride<33554432, 32768, 512, 32, 1>, pto::Layout::NZ>(v1635 + (v1636 + v1534 * v1637), v1638, v1639);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v1633, v1640);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %stage_5_v32__ssa_v0
      ;
      Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v1641 = Tile<TileType::Right, int8_t, 128, 256, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v332, v329);
      // pto: %stage_5_v32__ssa_v0
      ;
      uint64_t v1642 = (uint64_t) v301;
      TASSIGN(v1641, v1642);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v1641, v1633);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL_ACC(v1557, v1557, v1615, v1627);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v1557, v1557, v1629, v1641);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      // pto: %687
      ;
      int64_t v1643 = v1526 < v350 ? v350 : v1526;
      // pto: %temporary_25__ssa_v0_pview
      ;
      __gm__ int32_t* v1644 = PTOAS__GLOBAL_TENSOR_DATA(v483);
      // pto: %temporary_25__ssa_v0_pview
      ;
      const int64_t v1645 = 0;
      // pto: %temporary_25__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 96, 256> v1646 = pto::Shape<1, 1, 1, 96, 256>();
      // pto: %temporary_25__ssa_v0_pview
      ;
      pto::Stride<3145728, 3145728, 3145728, 32768, 1> v1647 = pto::Stride<3145728, 3145728, 3145728, 32768, 1>();
      // pto: %temporary_25__ssa_v0_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 96, 256>, pto::Stride<3145728, 3145728, 3145728, 32768, 1>, pto::Layout::ND> v1648 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 96, 256>, pto::Stride<3145728, 3145728, 3145728, 32768, 1>, pto::Layout::ND>(v1644 + (v1645 + v1643), v1646, v1647);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
      TSTORE(v1648, v1557);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE2);
    pipe_barrier(PIPE_MTE3);
    pipe_barrier(PIPE_FIX);
    dsb(DSB_DDR);
    dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
    pipe_barrier(PIPE_MTE2);
    pipe_barrier(PIPE_MTE3);
    pipe_barrier(PIPE_FIX);
    dsb(DSB_DDR);
    // pto: %688
    ;
    if (v1505 != v292) {
      pipe_barrier(PIPE_ALL);
      // pto: %692, %693
      ;
      int32_t v1649 = (int32_t) ((int64_t) (uint64_t) v853 + (uint64_t) v1502);
      pto::comm::NotifyOp v1650 = pto::comm::NotifyOp::Set;
      pto::comm::TNOTIFY(v850, v1649, v1650);
    };
  }
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  // pto: %695, %696
  int64_t v1651 = (v131)[(int64_t) ((uint64_t) v856 + (uint64_t) v364)];
  // pto: %698
  // pto: %699
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  if (v1651 >= v350) {
    // pto: %done_6__tile
    ;
    int64_t v1652 = (v131)[(int64_t) ((uint64_t) v856 + (uint64_t) v269)];
    // pto: %703
    ;
    int64_t v1653 = (v131)[(int64_t) ((uint64_t) v856 + (uint64_t) v268)];
    // pto: %706
    ;
    int64_t v1654 = (v131)[(int64_t) ((uint64_t) v856 + (uint64_t) v267)];
    // pto: %publish_6__tile
    ;
    int64_t v1655 = (v131)[(int64_t) ((uint64_t) v856 + (uint64_t) v266)];
    for (int64_t v1656 = v350; v1656 < v1654; v1656 += v317) {
      // pto: %710, %711
      ;
      int64_t v1657 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v1653 + (uint64_t) v1656) * (uint64_t) v335);
      // pto: %712
      ;
      int64_t v1658 = (v132)[v1657];
      // pto: %714, %required_6__tile
      ;
      int64_t v1659 = (v132)[(int64_t) ((uint64_t) v1657 + (uint64_t) v317)];
      // pto: %717
      ;
      int64_t v1660 = (int64_t) ((uint64_t) v1658 * (uint64_t) v332);
      // pto: %718
      ;
      int64_t v1661 = v1660 < v350 ? v350 : v1660;
      // pto: %719
      ;
      __gm__ int32_t* v1662 = PTOAS__GLOBAL_TENSOR_DATA(v766);
      // pto: %719
      ;
      const int64_t v1663 = 0;
      // pto: %719
      ;
      pto::Shape<1, 1, 1, 1, 1> v1664 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %719
      ;
      pto::Stride<1, 1, 1, 1, 1> v1665 = pto::Stride<1, 1, 1, 1, 1>();
      // pto: %719
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::ND> v1666 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::ND>(v1662 + (v1663 + v1661), v1664, v1665);
      // pto: %716, %720
      ;
      int32_t v1667 = (int32_t) ((int64_t) (uint64_t) v853 + (uint64_t) v1659);
      pto::comm::WaitCmp v1668 = pto::comm::WaitCmp::GE;
      pto::comm::TWAIT(v1666, v1667, v1668);
    };
    pipe_barrier(PIPE_MTE2);
    pipe_barrier(PIPE_MTE3);
    pipe_barrier(PIPE_FIX);
    dsb(DSB_DDR);
    dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
    pipe_barrier(PIPE_MTE2);
    pipe_barrier(PIPE_MTE3);
    pipe_barrier(PIPE_FIX);
    dsb(DSB_DDR);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v1669 = v1651 < v350 ? v350 : v1651; v1669 < v322; v1669 += v318) {
      // pto: %721
      ;
      int64_t v1670 = v1669 / v324;
      // pto: %723, %722
      ;
      int64_t v1671 = (int64_t) ((uint64_t) v1669 - (uint64_t) ((int64_t) (uint64_t) v1670 * (uint64_t) v324));
      // pto: %724
      ;
      int64_t v1672 = (int64_t) ((uint64_t) v1670 * (uint64_t) v325);
      // pto: %725
      ;
      int64_t v1673 = (int64_t) ((uint64_t) v316 - (uint64_t) v1672);
      // pto: %726
      ;
      int64_t v1674 = v1673 < v325 ? v1673 : v325;
      