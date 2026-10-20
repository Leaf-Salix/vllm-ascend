# pypto.program: _jit_entry
import pypto.language as pl

PAIRED_COMPACT_DIRECT_ROWS = pl.dynamic("PAIRED_COMPACT_DIRECT_ROWS")
t__tmp_v1 = pl.dynamic("t__tmp_v1")


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def compact_rows(
        src__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 2048], pl.FP32],
        valid__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.INT32],
        thresholds__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.FP32],
        offsets__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.INT32],
        values_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32]],
        ids_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32]],
        counts_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]]:
        t__tmp_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(src__ssa_v0, 0)
        for row__idx_v0, (counts_out__iter_v1,) in pl.range(t__tmp_v0, init_values=(counts_out__ssa_v0,)):
            values__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.load(src__ssa_v0, [row__idx_v0, 0], [1, 2048], [1, 2048])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid__ssa_v0, [row__idx_v0])
            t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(valid__ssa_v0, [row__idx_v0])
            values_v1__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.TileView(valid_shape=[1, pl.cast(t__tmp_v1, pl.INDEX)])] = pl.tile.set_validshape(values__ssa_v0, 1, pl.cast(t__tile_1, pl.INDEX))
            threshold__tile: pl.Scalar[pl.FP32] = pl.tensor.read(thresholds__ssa_v0, [row__idx_v0])
            offset__tile: pl.Scalar[pl.INT32] = pl.tensor.read(offsets__ssa_v0, [row__idx_v0])
            scratch__ssa_v0: pl.Tile[[1, 18432], pl.UINT8] = pl.tile.create([1, 18432], dtype=pl.UINT8)
            _tuple_tmp__ssa_v0: pl.Tuple[pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec], pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec], pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 1])]] = (
                pl.tile.compact_compare(values_v1__ssa_v0_1__FREE_VAR, threshold__tile, scratch__ssa_v0, offset__tile, cmp_mode=5, out_cols=1024)
            )
            packed__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = _tuple_tmp__ssa_v0[0]
            ids__ssa_v0: pl.Tile[[1, 1024], pl.INT32, pl.Mem.Vec] = _tuple_tmp__ssa_v0[1]
            count__ssa_v0: pl.Tile[[1, 8], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 1])] = _tuple_tmp__ssa_v0[2]
            pl.tile.store(packed__ssa_v0, [row__idx_v0, 8], values_out__ssa_v0)
            pl.tile.store(ids__ssa_v0, [row__idx_v0, 8], ids_out__ssa_v0)
            t__tmp_v3: pl.Scalar[pl.INT32] = pl.tile.read(count__ssa_v0, [0, 0])
            counts_out__tile: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32] = pl.tensor.write(counts_out__iter_v1, [row__idx_v0, 8], t__tmp_v3)
            counts_out__rv_v2: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32] = pl.yield_(counts_out__tile)
        return values_out__ssa_v0, ids_out__ssa_v0, counts_out__rv_v2

    @pl.function(type=pl.FunctionType.Spmd)
    def entry_spmd_0(
        self,
        src__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 2048], pl.FP32],
        valid__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.INT32],
        thresholds__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.FP32],
        offsets__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.INT32],
        values_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32]],
        ids_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32]],
        counts_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]]:
        _tuple_tmp__ssa_v0: pl.Tuple[
            pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]
        ] = self.compact_rows(src__ssa_v0, valid__ssa_v0, thresholds__ssa_v0, offsets__ssa_v0, values_out__ssa_v0, ids_out__ssa_v0, counts_out__ssa_v0)
        values_out__ssa_v1: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32] = _tuple_tmp__ssa_v0[0]
        ids_out__ssa_v1: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32] = _tuple_tmp__ssa_v0[1]
        counts_out__ssa_v1: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32] = _tuple_tmp__ssa_v0[2]
        return values_out__ssa_v0, ids_out__ssa_v0, counts_out__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry(
        self,
        src__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 2048], pl.FP32],
        valid__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.INT32],
        thresholds__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.FP32],
        offsets__ssa_v0: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS], pl.INT32],
        values_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32]],
        ids_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32]],
        counts_out__ssa_v0: pl.InOut[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]]:
        ret__tmp_v0: pl.Tuple[
            pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32], pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32]
        ] = self.entry_spmd_0(src__ssa_v0, valid__ssa_v0, thresholds__ssa_v0, offsets__ssa_v0, values_out__ssa_v0, ids_out__ssa_v0, counts_out__ssa_v0, attrs={"core_num": 1})
        values_out__ssa_v1: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.FP32] = ret__tmp_v0[0]
        ids_out__ssa_v1: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 1040], pl.INT32] = ret__tmp_v0[1]
        counts_out__ssa_v1: pl.Tensor[[PAIRED_COMPACT_DIRECT_ROWS, 16], pl.INT32] = ret__tmp_v0[2]
        return values_out__ssa_v1, ids_out__ssa_v1, counts_out__ssa_v1
