补充一个相近的**循环回边**复现，便于把这种访问模式纳入修复范围。此处是同步分析保守带来的优化缺失，没有发现数值错误。我只验证了现有的 stock 0.66 / 0.67 二进制，**没有验证上面提到的 GitCode PR #290 分支**，不能据此判断该 PR 是否已经覆盖。

每轮只写 dense ND UB 父 Tile 中的一行：偶数轮由 MTE2 写，奇数轮由 V 写。循环范围 0..63、步长 1，每行 1024 B，写入区间 `[1024*r, 1024*(r+1))` 两两不交；初始化及最终整块写回仍有真实依赖。

```bash
ptoas --version
ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync disjoint_rows.pto -o disjoint_rows.cpp
```

完整输入：

```mlir
// CPU-only InsertSync reproducer: each iteration writes one disjoint UB row.
!parent = !pto.tile_buf<loc=vec, dtype=bf16, rows=64, cols=512, v_row=64, v_col=512, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!row = !pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=512, v_row=1, v_col=512, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!src = !pto.partition_tensor_view<1x512xbf16>
!out = !pto.partition_tensor_view<64x512xbf16>
module {
  func.func @disjoint_rows(%src: !src, %out: !out) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %a0 = arith.constant 0 : i64
    %a65536 = arith.constant 65536 : i64
    %zero = arith.constant 0.0 : bf16
    %parent = pto.alloc_tile addr = %a0 : !parent
    %zero_row = pto.alloc_tile addr = %a65536 : !row
    pto.texpands ins(%zero : bf16) outs(%parent : !parent)
    pto.texpands ins(%zero : bf16) outs(%zero_row : !row)
    scf.for %r = %c0 to %c64 step %c1 {
      %target = pto.subview %parent[%r, %c0] sizes [1, 512] : !parent -> !row
      %parity = arith.remui %r, %c2 : index
      %take_dma = arith.cmpi eq, %parity, %c0 : index
      scf.if %take_dma {
        pto.tload ins(%src : !src) outs(%target : !row)
      } else {
        pto.tmov ins(%zero_row : !row) outs(%target : !row)
      }
    }
    pto.tstore ins(%parent : !parent) outs(%out : !out)
    return
  }
}
```

0.66 和 0.67 均编译通过，但生成循环每轮都含：

```cpp
wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
if (r % 2 == 0) {
  TLOAD(row_r, source);
} else {
  pipe_barrier(PIPE_V);
  TMOV(row_r, zero_row);
}
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
```

同址负对照：只把 `pto.subview %parent[%r, %c0]` 改为 `pto.subview %parent[%c0, %c0]`。两版本也都编译通过并保留相同回边；这个对照中的跨迭代覆盖依赖应当保留。

建议在现有 subview/slot 描述符基础上，为有界、正单位步长的 dense ND 行访问补充跨迭代区域证明：只有已证明第 i−1 轮和第 i 轮的读写区域不相交时，才去掉相应回边。这里不需要修改全局 MemAlias 规则。必须保留初始化 V→MTE2、结束时 V/MTE2→MTE3、同址/未知区域及环形复用时的实际依赖；MTE3 若为最后读者，也仍须保护其完成后复用。

建议回归包含 0/1/N 次迭代、两分支、同址负对照、未知偏移以及带 MTE3 读者的版本。当前证据是完整编译输入和生成同步位置；未把这个 CPU 缩小例作为设备性能提升证据。在 CSA 中将合法 DMA 和无效补零拆成两遍可避开逐行事件，但完整算子仍未超过现有 MGATHER 版本，因此也没有宣称解决此分析问题就必然达到某个性能收益。
