# pypto.program: _jit_entry
import pypto.language as pl


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def oa_pair_explicit_mat(
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32]],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        group__ssa_v0: pl.Scalar[pl.INDEX] = worker__ssa_v0 // 3
        pair__ssa_v0: pl.Scalar[pl.INDEX] = worker__ssa_v0 % 3
        row_base__ssa_v0: pl.Scalar[pl.INDEX] = group__ssa_v0 * 384
        n0__ssa_v0: pl.Scalar[pl.INDEX] = pair__ssa_v0 * 256
        output_col__ssa_v0: pl.Scalar[pl.INDEX] = group__ssa_v0 * 1024 + n0__ssa_v0
        seed0__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        seed1__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        valid0__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(
            seed0__ssa_v0, valid_rows__ssa_v0, 128
        )
        valid1__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(
            seed1__ssa_v0, valid_rows__ssa_v0, 128
        )
        a_seed__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
            lhs__ssa_v0, [row_base__ssa_v0, 0], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
        )
        w0_seed__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
            rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, 0, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
        )
        w1_seed__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
            rhs__ssa_v0, [group__ssa_v0, (n0__ssa_v0 + 128) // 16, 0, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
        )
        for outer__idx_v0, (a_cur, w0_cur, w1_cur, acc0, acc1) in pl.range(15, init_values=(a_seed__ssa_v0, w0_seed__ssa_v0, w1_seed__ssa_v0, valid0__ssa_v0, valid1__ssa_v0)):
            next_k__ssa_v0: pl.Scalar[pl.INDEX] = outer__idx_v0 * 256 + 256
            a_next__ssa_v0: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows__ssa_v0, 256])] = pl.tile.load(
                lhs__ssa_v0, [row_base__ssa_v0, next_k__ssa_v0], [128, 256], [valid_rows__ssa_v0, 256], target_memory=pl.Mem.Mat
            )
            w0_next__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, n0__ssa_v0 // 16, next_k__ssa_v0 // 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w1_next__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                rhs__ssa_v0, [group__ssa_v0, (n0__ssa_v0 + 128) // 16, next_k__ssa_v0 // 16, 0, 0], [1, 8, 16, 16, 16], [1, 8, 16, 16, 16], target_memory=pl.Mem.Mat, cache=1
            )
            w0_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w0_cur, [256, 128])
            w1_flat__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w1_cur, [256, 128])
            left0__ssa_v0: pl.Tile[
                [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
            ] = pl.tile.extract(a_cur, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            right0a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            right0b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1_flat__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            acc0_first__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                acc0, left0__ssa_v0, right0a__ssa_v0, outer__idx_v0 == 0
            )
            left1__ssa_v0: pl.Tile[
                [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
            ] = pl.tile.extract(a_cur, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            right1a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            acc1_first__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                acc1, left0__ssa_v0, right0b__ssa_v0, outer__idx_v0 == 0
            )
            right1b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1_flat__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            acc0_last__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                acc0_first__ssa_v0, left1__ssa_v0, right1a__ssa_v0, pl.const(0, pl.BOOL)
            )
            acc1_last__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                acc1_first__ssa_v0, left1__ssa_v0, right1b__ssa_v0, pl.const(0, pl.BOOL)
            )
            last_a, last_w0, last_w1, done0, done1 = pl.yield_(a_next__ssa_v0, w0_next__ssa_v0, w1_next__ssa_v0, acc0_last__ssa_v0, acc1_last__ssa_v0)
        tail_w0__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(last_w0, [256, 128])
        tail_w1__ssa_v0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(last_w1, [256, 128])
        tail_l0__ssa_v0: pl.Tile[
            [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
        ] = pl.tile.extract(last_a, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        tail_r0a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w0__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right)
        tail_r0b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w1__ssa_v0, 0, 0, [128, 128], target_memory=pl.Mem.Right)
        tail_acc0_first__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
            done0, tail_l0__ssa_v0, tail_r0a__ssa_v0, pl.const(0, pl.BOOL)
        )
        tail_l1__ssa_v0: pl.Tile[
            [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows__ssa_v0, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
        ] = pl.tile.extract(last_a, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        tail_r1a__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w0__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right)
        tail_acc1_first__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
            done1, tail_l0__ssa_v0, tail_r0b__ssa_v0, pl.const(0, pl.BOOL)
        )
        tail_r1b__ssa_v0: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w1__ssa_v0, 128, 0, [128, 128], target_memory=pl.Mem.Right)
        final0__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
            tail_acc0_first__ssa_v0, tail_l1__ssa_v0, tail_r1a__ssa_v0, pl.const(0, pl.BOOL)
        )
        final1__ssa_v0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
            tail_acc1_first__ssa_v0, tail_l1__ssa_v0, tail_r1b__ssa_v0, pl.const(0, pl.BOOL)
        )
        output__store: pl.Tensor[[384, 8192], pl.FP32] = pl.tile.store(final0__ssa_v0, [0, output_col__ssa_v0], output__ssa_v0)
        output__store_v0: pl.Tensor[[384, 8192], pl.FP32] = pl.tile.store(final1__ssa_v0, [0, output_col__ssa_v0 + 128], output__ssa_v0)
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def oa_pair_explicit_mat_spmd(
        self,
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32]],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        output__ssa_v1: pl.Tensor[[384, 8192], pl.FP32] = self.oa_pair_explicit_mat(valid_rows__ssa_v0, lhs__ssa_v0, rhs__ssa_v0, output__ssa_v0)
        return output__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(
        self,
        lhs__ssa_v0: pl.Tensor[[3072, 4096], pl.BF16],
        rhs__ssa_v0: pl.Tensor[[8, 64, 256, 16, 16], pl.BF16, pl.TensorView(stride=[4194304, 65536, 256, 16, 1], layout=pl.TensorLayout.NZ)],
        output__ssa_v0: pl.Out[pl.Tensor[[384, 8192], pl.FP32]],
        valid_rows__ssa_v0: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        pl.func_attr({"nz_tensor_views_blocked": True, "mx_tensor_views_blocked": True})
        output__ssa_v1: pl.Tensor[[384, 8192], pl.FP32] = self.oa_pair_explicit_mat_spmd(valid_rows__ssa_v0, lhs__ssa_v0, rhs__ssa_v0, output__ssa_v0, attrs={"core_num": 24})
        return output__ssa_v1
