# pypto.program: _jit_entry
import pypto.language as pl


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(
        self,
        lhs: pl.Tensor[[3072, 4096], pl.BF16],
        rhs: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)],
        output: pl.Out[pl.Tensor[[384, 8192], pl.FP32]],
        valid_rows: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[384, 8192], pl.FP32]:
        for worker in pl.spmd(24, name_hint="oa_pair_explicit_mat_spmd"):
            group: pl.Scalar[pl.INDEX] = worker // 3
            pair: pl.Scalar[pl.INDEX] = worker % 3
            row_base: pl.Scalar[pl.INDEX] = group * 384
            n0: pl.Scalar[pl.INDEX] = pair * 256
            output_col: pl.Scalar[pl.INDEX] = group * 1024 + n0
            seed0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            seed1: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            valid0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(seed0, valid_rows, 128)
            valid1: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(seed1, valid_rows, 128)
            a_seed: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(lhs, [row_base, 0], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w0_seed: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, 0, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w1_seed: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, 0, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            for outer, (a_cur, w0_cur, w1_cur, acc0, acc1) in pl.range(15, init_values=(a_seed, w0_seed, w1_seed, valid0, valid1)):
                next_k: pl.Scalar[pl.INDEX] = (outer + 1) * 256
                a_next: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(
                    lhs, [row_base, next_k], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat
                )
                w0_next: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, next_k, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w1_next: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, next_k, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w0_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w0_cur, [256, 128])
                w1_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w1_cur, [256, 128])
                left0: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a_cur, 0, 0, [128, 128], target_memory=pl.Mem.Left)
                right0a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                right0b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                acc0_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(acc0, left0, right0a, outer == 0)
                left1: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a_cur, 0, 128, [128, 128], target_memory=pl.Mem.Left)
                right1a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                acc1_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(acc1, left0, right0b, outer == 0)
                right1b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                acc0_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    acc0_first, left1, right1a, pl.const(0, pl.BOOL)
                )
                acc1_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    acc1_first, left1, right1b, pl.const(0, pl.BOOL)
                )
                last_a, last_w0, last_w1, done0, done1 = pl.yield_(a_next, w0_next, w1_next, acc0_last, acc1_last)
            tail_w0: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(last_w0, [256, 128])
            tail_w1: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(last_w1, [256, 128])
            tail_l0: pl.Tile[
                [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
            ] = pl.tile.extract(last_a, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            tail_r0a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w0, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            tail_r0b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w1, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            tail_acc0_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                done0, tail_l0, tail_r0a, pl.const(0, pl.BOOL)
            )
            tail_l1: pl.Tile[
                [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
            ] = pl.tile.extract(last_a, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            tail_r1a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w0, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            tail_acc1_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                done1, tail_l0, tail_r0b, pl.const(0, pl.BOOL)
            )
            tail_r1b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(tail_w1, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            final0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                tail_acc0_first, tail_l1, tail_r1a, pl.const(0, pl.BOOL)
            )
            final1: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                tail_acc1_first, tail_l1, tail_r1b, pl.const(0, pl.BOOL)
            )
            pl.tile.store(final0, [0, output_col], output)
            pl.tile.store(final1, [0, output_col + 128], output)
        return output
