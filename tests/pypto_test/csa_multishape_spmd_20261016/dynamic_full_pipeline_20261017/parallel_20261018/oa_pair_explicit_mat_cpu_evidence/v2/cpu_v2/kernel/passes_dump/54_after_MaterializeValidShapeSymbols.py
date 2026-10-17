# pypto.program: _jit_entry
import pypto.language as pl


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def oa_pair_four_k256(
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_acc_4: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_11: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_12: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_right_13: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_14: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        group__ssa_v0: pl.Scalar[pl.INDEX] = worker__ssa_v0 // 3
        pair__ssa_v0: pl.Scalar[pl.INDEX] = worker__ssa_v0 % 3
        row_base__ssa_v0: pl.Scalar[pl.INDEX] = group__ssa_v0 * 384
        n0__ssa_v0: pl.Scalar[pl.INDEX] = pair__ssa_v0 * 256
        output_col__ssa_v0: pl.Scalar[pl.INDEX] = group__ssa_v0 * 1024 + n0__ssa_v0
        seed0__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create(
            [128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True
        )
        seed1__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create(
            [128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True
        )
        valid0__ssa_v0: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.set_validshape(seed0__ssa_v0, valid_rows__ssa_v0, 128)
        valid1__ssa_v0: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.set_validshape(seed1__ssa_v0, valid_rows__ssa_v0, 128)
        for outer__idx_v0, (acc0, acc1) in pl.range(4, init_values=(valid0__ssa_v0, valid1__ssa_v0)):
            kbase__ssa_v0: pl.Scalar[pl.INDEX] = outer__idx_v0 * 1024
            a0__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
                lhs__ssa_v0, [row_base__ssa_v0, kbase__ssa_v0], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            w0a__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, kbase__ssa_v0 // 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w0b__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16 + 8, kbase__ssa_v0 // 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            a1__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_8, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
                lhs__ssa_v0, [row_base__ssa_v0, kbase__ssa_v0 + 256], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            w1a__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, kbase__ssa_v0 // 16 + 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w1b__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16 + 8, kbase__ssa_v0 // 16 + 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w0a_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = w0a__ssa_v0
            w0b_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = w0b__ssa_v0
            l00__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a0__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r00a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w0a_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            r00b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w0b_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c0a_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc0, l00__ssa_v0, r00a__ssa_v0, outer__idx_v0 == 0)
            l01__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_14, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a0__ssa_v0, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r01a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w0a_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c0b_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc1, l00__ssa_v0, r00b__ssa_v0, outer__idx_v0 == 0)
            r01b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w0b_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c0a_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c0a_first__ssa_v0, l01__ssa_v0, r01a__ssa_v0, pl.const(0, pl.BOOL))
            c0b_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c0b_first__ssa_v0, l01__ssa_v0, r01b__ssa_v0, pl.const(0, pl.BOOL))
            a2__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
                lhs__ssa_v0, [row_base__ssa_v0, kbase__ssa_v0 + 512], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            w2a__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, kbase__ssa_v0 // 16 + 32, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w2b__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16 + 8, kbase__ssa_v0 // 16 + 32, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w1a_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat] = w1a__ssa_v0
            w1b_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = w1b__ssa_v0
            l10__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a1__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r10a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w1a_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            r10b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w1b_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c1a_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c0a_last__ssa_v0, l10__ssa_v0, r10a__ssa_v0, pl.const(0, pl.BOOL))
            l11__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_14, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a1__ssa_v0, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r11a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w1a_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c1b_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c0b_last__ssa_v0, l10__ssa_v0, r10b__ssa_v0, pl.const(0, pl.BOOL))
            r11b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w1b_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c1a_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c1a_first__ssa_v0, l11__ssa_v0, r11a__ssa_v0, pl.const(0, pl.BOOL))
            c1b_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c1b_first__ssa_v0, l11__ssa_v0, r11b__ssa_v0, pl.const(0, pl.BOOL))
            a3__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_8, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
                lhs__ssa_v0, [row_base__ssa_v0, kbase__ssa_v0 + 768], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            w3a__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, kbase__ssa_v0 // 16 + 48, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w3b__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16 + 8, kbase__ssa_v0 // 16 + 48, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w2a_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = w2a__ssa_v0
            w2b_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = w2b__ssa_v0
            l20__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a2__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r20a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w2a_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            r20b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w2b_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c2a_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c1a_last__ssa_v0, l20__ssa_v0, r20a__ssa_v0, pl.const(0, pl.BOOL))
            l21__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_14, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a2__ssa_v0, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r21a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w2a_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c2b_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c1b_last__ssa_v0, l20__ssa_v0, r20b__ssa_v0, pl.const(0, pl.BOOL))
            r21b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w2b_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c2a_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c2a_first__ssa_v0, l21__ssa_v0, r21a__ssa_v0, pl.const(0, pl.BOOL))
            c2b_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c2b_first__ssa_v0, l21__ssa_v0, r21b__ssa_v0, pl.const(0, pl.BOOL))
            w3a_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat] = w3a__ssa_v0
            w3b_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = w3b__ssa_v0
            l30__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a3__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r30a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w3a_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            r30b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w3b_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c3a_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c2a_last__ssa_v0, l30__ssa_v0, r30a__ssa_v0, pl.const(0, pl.BOOL))
            l31__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_14, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a3__ssa_v0, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r31a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w3a_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c3b_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c2b_last__ssa_v0, l30__ssa_v0, r30b__ssa_v0, pl.const(0, pl.BOOL))
            r31b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w3b_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            c3a_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c3a_first__ssa_v0, l31__ssa_v0, r31a__ssa_v0, pl.const(0, pl.BOOL))
            c3b_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(c3b_first__ssa_v0, l31__ssa_v0, r31b__ssa_v0, pl.const(0, pl.BOOL))
            done0, done1 = pl.yield_(c3a_last__ssa_v0, c3b_last__ssa_v0)
        output__store: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(done0, [0, output_col__ssa_v0], output__ssa_v0)
        output__store_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(done1, [0, output_col__ssa_v0 + 128], output__ssa_v0)
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def oa_pair_four_k256_spmd(
        self,
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        output__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.oa_pair_four_k256(
            valid_rows__ssa_v0, lhs__ssa_v0, rhs__ssa_v0, output__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing]}
        )
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def entry(
        self,
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        with pl.scope():
            output__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.oa_pair_four_k256_spmd(
                valid_rows__ssa_v0, lhs__ssa_v0, rhs__ssa_v0, output__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing], "core_num": 24}
            )
            return output__ssa_v1
