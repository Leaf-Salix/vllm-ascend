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
        for worker in pl.spmd(24, name_hint="oa_pair_four_k256_spmd"):
            group: pl.Scalar[pl.INDEX] = worker // 3
            pair: pl.Scalar[pl.INDEX] = worker % 3
            row_base: pl.Scalar[pl.INDEX] = group * 384
            n0: pl.Scalar[pl.INDEX] = pair * 256
            output_col: pl.Scalar[pl.INDEX] = group * 1024 + n0
            seed0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            seed1: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
            valid0: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(seed0, valid_rows, 128)
            valid1: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(seed1, valid_rows, 128)
            for outer, (acc0, acc1) in pl.range(4, init_values=(valid0, valid1)):
                kbase: pl.Scalar[pl.INDEX] = outer * 1024
                a0: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(
                    lhs, [row_base, kbase + 0], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat
                )
                w0a: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 0, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w0b: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 0, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                a1: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(
                    lhs, [row_base, kbase + 256], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat
                )
                w1a: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 256, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w1b: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 256, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w0a_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w0a, [256, 128])
                w0b_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w0b, [256, 128])
                l00: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a0, 0, 0, [128, 128], target_memory=pl.Mem.Left)
                r00a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                r00b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                c0a_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(acc0, l00, r00a, outer == 0)
                l01: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a0, 0, 128, [128, 128], target_memory=pl.Mem.Left)
                r01a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c0b_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(acc1, l00, r00b, outer == 0)
                r01b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w0b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c0a_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c0a_first, l01, r01a, pl.const(0, pl.BOOL)
                )
                c0b_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c0b_first, l01, r01b, pl.const(0, pl.BOOL)
                )
                a2: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(
                    lhs, [row_base, kbase + 512], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat
                )
                w2a: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 512, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w2b: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 512, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w1a_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w1a, [256, 128])
                w1b_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w1b, [256, 128])
                l10: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
                r10a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                r10b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                c1a_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c0a_last, l10, r10a, pl.const(0, pl.BOOL)
                )
                l11: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
                r11a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c1b_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c0b_last, l10, r10b, pl.const(0, pl.BOOL)
                )
                r11b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w1b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c1a_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c1a_first, l11, r11a, pl.const(0, pl.BOOL)
                )
                c1b_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c1b_first, l11, r11b, pl.const(0, pl.BOOL)
                )
                a3: pl.Tile[[128, 256], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(
                    lhs, [row_base, kbase + 768], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat
                )
                w3a: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 768, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w3b: pl.Tile[[1, 256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(rhs, [group, kbase + 768, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
                w2a_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w2a, [256, 128])
                w2b_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w2b, [256, 128])
                l20: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
                r20a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w2a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                r20b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w2b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                c2a_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c1a_last, l20, r20a, pl.const(0, pl.BOOL)
                )
                l21: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
                r21a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w2a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c2b_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c1b_last, l20, r20b, pl.const(0, pl.BOOL)
                )
                r21b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w2b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c2a_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c2a_first, l21, r21a, pl.const(0, pl.BOOL)
                )
                c2b_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c2b_first, l21, r21b, pl.const(0, pl.BOOL)
                )
                w3a_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w3a, [256, 128])
                w3b_flat: pl.Tile[[256, 128], pl.BF16, pl.Mem.Mat] = pl.tile.reshape(w3b, [256, 128])
                l30: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a3, 0, 0, [128, 128], target_memory=pl.Mem.Left)
                r30a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w3a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                r30b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w3b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                c3a_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c2a_last, l30, r30a, pl.const(0, pl.BOOL)
                )
                l31: pl.Tile[
                    [128, 128], pl.BF16, pl.Mem.Left, pl.TileView(valid_shape=[pl.min(valid_rows, pl.const(128, pl.INDEX)), 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)
                ] = pl.tile.extract(a3, 0, 128, [128, 128], target_memory=pl.Mem.Left)
                r31a: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w3a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c3b_first: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c2b_last, l30, r30b, pl.const(0, pl.BOOL)
                )
                r31b: pl.Tile[[128, 128], pl.BF16, pl.Mem.Right] = pl.tile.extract(w3b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                c3a_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c3a_first, l31, r31a, pl.const(0, pl.BOOL)
                )
                c3b_last: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[valid_rows, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(
                    c3b_first, l31, r31b, pl.const(0, pl.BOOL)
                )
                done0, done1 = pl.yield_(c3a_last, c3b_last)
            pl.tile.store(done0, [0, output_col], output)
            pl.tile.store(done1, [0, output_col + 128], output)
        return output
