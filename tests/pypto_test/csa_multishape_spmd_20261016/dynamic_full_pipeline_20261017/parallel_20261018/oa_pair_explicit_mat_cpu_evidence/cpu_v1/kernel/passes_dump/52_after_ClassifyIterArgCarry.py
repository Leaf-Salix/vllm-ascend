# pypto.program: _jit_entry
import pypto.language as pl


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def oa_pair_explicit_mat(
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
        a_seed__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
            lhs__ssa_v0, [row_base__ssa_v0, 0], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
        )
        w0_seed__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, 0, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
        )
        w1_seed__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
            rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16 + 8, 0, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
        )
        for outer__idx_v0, (a_cur, w0_cur, w1_cur, acc0, acc1) in pl.range(15, init_values=(a_seed__ssa_v0, w0_seed__ssa_v0, w1_seed__ssa_v0, valid0__ssa_v0, valid1__ssa_v0)):
            next_k__ssa_v0: pl.Scalar[pl.INDEX] = outer__idx_v0 * 256 + 256
            a_next__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_8, pl.const(262144, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
                lhs__ssa_v0, [row_base__ssa_v0, next_k__ssa_v0], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            w0_next__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_9, pl.const(327680, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, next_k__ssa_v0 // 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w1_next__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16 + 8, next_k__ssa_v0 // 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w0_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = w0_cur
            w1_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = w1_cur
            left0__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a_cur, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            right0a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w0_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            right0b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w1_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc0_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc0, left0__ssa_v0, right0a__ssa_v0, outer__idx_v0 == 0)
            left1__ssa_v0: pl.Tile[
                [128, 128],
                pl.BF16,
                pl.MemRef(mem_left_14, pl.const(32768, pl.INT64), 32768),
                pl.Mem.Left,
                pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
            ] = pl.tile.extract(a_cur, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            right1a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w0_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc1_first__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc1, left0__ssa_v0, right0b__ssa_v0, outer__idx_v0 == 0)
            right1b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
                w1_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
            )
            acc0_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc0_first__ssa_v0, left1__ssa_v0, right1a__ssa_v0, pl.const(0, pl.BOOL))
            acc1_last__ssa_v0: pl.Tile[
                [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
            ] = pl.tile.matmul_acc(acc1_first__ssa_v0, left1__ssa_v0, right1b__ssa_v0, pl.const(0, pl.BOOL))
            w1_next__ssa_v0_mv: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.move(w1_next__ssa_v0, target_memory=pl.Mem.Mat)
            w0_next__ssa_v0_mv: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.move(w0_next__ssa_v0, target_memory=pl.Mem.Mat)
            a_next__ssa_v0_mv: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_5, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.move(
                a_next__ssa_v0, target_memory=pl.Mem.Mat
            )
            last_a, last_w0, last_w1, done0, done1 = pl.yield_(a_next__ssa_v0_mv, w0_next__ssa_v0_mv, w1_next__ssa_v0_mv, acc0_last__ssa_v0, acc1_last__ssa_v0)
        tail_w0__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_6, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat] = last_w0
        tail_w1__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = last_w1
        tail_l0__ssa_v0: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(last_a, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        tail_r0a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            tail_w0__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        tail_r0b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            tail_w1__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right
        )
        tail_acc0_first__ssa_v0: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(done0, tail_l0__ssa_v0, tail_r0a__ssa_v0, pl.const(0, pl.BOOL))
        tail_l1__ssa_v0: pl.Tile[
            [128, 128],
            pl.BF16,
            pl.MemRef(mem_left_14, pl.const(32768, pl.INT64), 32768),
            pl.Mem.Left,
            pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal),
        ] = pl.tile.extract(last_a, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        tail_r1a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            tail_w0__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        tail_acc1_first__ssa_v0: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(done1, tail_l0__ssa_v0, tail_r0b__ssa_v0, pl.const(0, pl.BOOL))
        tail_r1b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_13, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(
            tail_w1__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right
        )
        final0__ssa_v0: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(tail_acc0_first__ssa_v0, tail_l1__ssa_v0, tail_r1a__ssa_v0, pl.const(0, pl.BOOL))
        final1__ssa_v0: pl.Tile[
            [128, 128], pl.FP32, pl.MemRef(mem_acc_4, pl.const(65536, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)
        ] = pl.tile.matmul_acc(tail_acc1_first__ssa_v0, tail_l1__ssa_v0, tail_r1b__ssa_v0, pl.const(0, pl.BOOL))
        output__store: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(final0__ssa_v0, [0, output_col__ssa_v0], output__ssa_v0)
        output__store_v0: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)] = pl.tile.store(final1__ssa_v0, [0, output_col__ssa_v0 + 128], output__ssa_v0)
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def oa_pair_explicit_mat_spmd(
        self,
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 25165824)],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ), pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 67108864)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12582912)]],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        output__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.oa_pair_explicit_mat(
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
            output__ssa_v1: pl.Tensor[[384, 8192], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 12582912)] = self.oa_pair_explicit_mat_spmd(
                valid_rows__ssa_v0, lhs__ssa_v0, rhs__ssa_v0, output__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing], "core_num": 24}
            )
            return output__ssa_v1
