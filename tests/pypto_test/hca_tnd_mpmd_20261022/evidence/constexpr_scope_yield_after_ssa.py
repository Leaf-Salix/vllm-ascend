# pypto.program: _jit_entry
import pypto.language as pl

T = pl.dynamic("T")


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def entry(self, x__ssa_v0: pl.Tensor[[T, 32], pl.FP32], out__ssa_v0: pl.Out[pl.Tensor[[T, 32], pl.FP32]]) -> pl.Tensor[[T, 32], pl.FP32]:
        if T <= 32:
            tokens_inline17__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
            tmp_inline15__ssa_v0: pl.Tensor[[tokens_inline17__ssa_v0, 32], pl.FP32] = pl.tensor.create([tokens_inline17__ssa_v0, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            deps_inline16__ssa_v0: pl.Array[1, pl.TASK_ID] = pl.array.create(1, dtype=pl.TASK_ID)
            if pl.const(1, pl.INDEX) == pl.const(1, pl.INDEX):
                with pl.spmd(1) as ready_inline39__ssa_v0:
                    row_inline38__ssa_v0: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX) * pl.const(32, pl.INDEX) + pl.tile.get_block_idx() * 32
                    count_inline37__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, pl.tensor.dim(x__ssa_v0, 0) - row_inline38__ssa_v0)
                    value_inline36__ssa_v0: pl.Tile[[32, 32], pl.FP32, pl.TileView(valid_shape=[count_inline37__ssa_v0, 32])] = pl.tile.load(
                        x__ssa_v0, [row_inline38__ssa_v0, 0], [32, 32], [count_inline37__ssa_v0, 32]
                    )
                    pl.tile.store(pl.tile.adds(value_inline36__ssa_v0, 1.0), [row_inline38__ssa_v0, 0], tmp_inline15__ssa_v0)
                tmp_inline15__ssa_v1: pl.Tensor[[tokens_inline17__ssa_v0, 32], pl.FP32] = tmp_inline15__ssa_v0
                ready_inline13__ssa_v0: pl.Scalar[pl.TASK_ID] = ready_inline39__ssa_v0
                deps_inline16__ssa_v1: pl.Array[1, pl.TASK_ID] = pl.array.update_element(deps_inline16__ssa_v0, 0, ready_inline13__ssa_v0)
                deps_inline16__phi_v5, tmp_inline15__phi_v5 = pl.yield_(deps_inline16__ssa_v1, tmp_inline15__ssa_v1)
            else:
                with pl.scope(mode=pl.ScopeMode.MANUAL):
                    for group_inline12__idx_v0, (deps_inline16__iter_v2, tmp_inline15__iter_v2) in pl.parallel(1, init_values=(deps_inline16__ssa_v0, tmp_inline15__ssa_v0)):
                        with pl.spmd(1) as ready_inline43__ssa_v0:
                            row_inline42__ssa_v0: pl.Scalar[pl.INDEX] = group_inline12__idx_v0 * 32 + pl.tile.get_block_idx() * 32
                            count_inline41__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, pl.tensor.dim(x__ssa_v0, 0) - row_inline42__ssa_v0)
                            value_inline40__ssa_v0: pl.Tile[[32, 32], pl.FP32, pl.TileView(valid_shape=[count_inline41__ssa_v0, 32])] = pl.tile.load(
                                x__ssa_v0, [row_inline42__ssa_v0, 0], [32, 32], [count_inline41__ssa_v0, 32]
                            )
                            pl.tile.store(pl.tile.adds(value_inline40__ssa_v0, 1.0), [row_inline42__ssa_v0, 0], tmp_inline15__iter_v2)
                        tmp_inline15__ssa_v4: pl.Tensor[[tokens_inline17__ssa_v0, 32], pl.FP32] = tmp_inline15__iter_v2
                        ready_inline13__ssa_v1: pl.Scalar[pl.TASK_ID] = ready_inline43__ssa_v0
                        deps_inline16__ssa_v4: pl.Array[1, pl.TASK_ID] = pl.array.update_element(deps_inline16__iter_v2, group_inline12__idx_v0, ready_inline13__ssa_v1)
                        deps_inline16__rv_v3, tmp_inline15__rv_v3 = pl.yield_(deps_inline16__ssa_v4, tmp_inline15__ssa_v4)
                    pl.yield_(deps_inline16__rv_v3, tmp_inline15__rv_v3)
            with pl.spmd(1, deps=[deps_inline16__phi_v5]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    row_inline10__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 32
                    count_inline9__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, tokens_inline17__ssa_v0 - row_inline10__ssa_v0)
                    value_inline8__ssa_v0: pl.Tile[[32, 32], pl.FP32, pl.TileView(valid_shape=[count_inline9__ssa_v0, 32])] = pl.tile.load(
                        tmp_inline15__phi_v5, [row_inline10__ssa_v0, 0], [32, 32], [count_inline9__ssa_v0, 32]
                    )
                    pl.tile.store(value_inline8__ssa_v0, [row_inline10__ssa_v0, 0], out__ssa_v0)
        else:
            tokens_inline27__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x__ssa_v0, 0)
            tmp_inline26__ssa_v0: pl.Tensor[[tokens_inline27__ssa_v0, 32], pl.FP32] = pl.tensor.create([tokens_inline27__ssa_v0, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            deps_inline23__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
            if pl.const(2, pl.INDEX) == pl.const(1, pl.INDEX):
                with pl.spmd(1) as ready_inline47__ssa_v0:
                    row_inline46__ssa_v0: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX) * pl.const(32, pl.INDEX) + pl.tile.get_block_idx() * 32
                    count_inline45__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, pl.tensor.dim(x__ssa_v0, 0) - row_inline46__ssa_v0)
                    value_inline44__ssa_v0: pl.Tile[[32, 32], pl.FP32, pl.TileView(valid_shape=[count_inline45__ssa_v0, 32])] = pl.tile.load(
                        x__ssa_v0, [row_inline46__ssa_v0, 0], [32, 32], [count_inline45__ssa_v0, 32]
                    )
                    pl.tile.store(pl.tile.adds(value_inline44__ssa_v0, 1.0), [row_inline46__ssa_v0, 0], tmp_inline26__ssa_v0)
                tmp_inline26__ssa_v1: pl.Tensor[[tokens_inline27__ssa_v0, 32], pl.FP32] = tmp_inline26__ssa_v0
                ready_inline21__ssa_v0: pl.Scalar[pl.TASK_ID] = ready_inline47__ssa_v0
                deps_inline23__ssa_v1: pl.Array[2, pl.TASK_ID] = pl.array.update_element(deps_inline23__ssa_v0, 0, ready_inline21__ssa_v0)
                deps_inline23__phi_v5, tmp_inline26__phi_v5 = pl.yield_(deps_inline23__ssa_v1, tmp_inline26__ssa_v1)
            else:
                with pl.scope(mode=pl.ScopeMode.MANUAL):
                    for group_inline19__idx_v0, (deps_inline23__iter_v2, tmp_inline26__iter_v2) in pl.parallel(2, init_values=(deps_inline23__ssa_v0, tmp_inline26__ssa_v0)):
                        with pl.spmd(1) as ready_inline51__ssa_v0:
                            row_inline50__ssa_v0: pl.Scalar[pl.INDEX] = group_inline19__idx_v0 * 32 + pl.tile.get_block_idx() * 32
                            count_inline49__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, pl.tensor.dim(x__ssa_v0, 0) - row_inline50__ssa_v0)
                            value_inline48__ssa_v0: pl.Tile[[32, 32], pl.FP32, pl.TileView(valid_shape=[count_inline49__ssa_v0, 32])] = pl.tile.load(
                                x__ssa_v0, [row_inline50__ssa_v0, 0], [32, 32], [count_inline49__ssa_v0, 32]
                            )
                            pl.tile.store(pl.tile.adds(value_inline48__ssa_v0, 1.0), [row_inline50__ssa_v0, 0], tmp_inline26__iter_v2)
                        tmp_inline26__ssa_v4: pl.Tensor[[tokens_inline27__ssa_v0, 32], pl.FP32] = tmp_inline26__iter_v2
                        ready_inline21__ssa_v1: pl.Scalar[pl.TASK_ID] = ready_inline51__ssa_v0
                        deps_inline23__ssa_v4: pl.Array[2, pl.TASK_ID] = pl.array.update_element(deps_inline23__iter_v2, group_inline19__idx_v0, ready_inline21__ssa_v1)
                        deps_inline23__rv_v3, tmp_inline26__rv_v3 = pl.yield_(deps_inline23__ssa_v4, tmp_inline26__ssa_v4)
                    pl.yield_(deps_inline23__rv_v3, tmp_inline26__rv_v3)
            with pl.spmd(2, deps=[deps_inline23__phi_v5]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    row_inline24__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 32
                    count_inline18__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, tokens_inline27__ssa_v0 - row_inline24__ssa_v0)
                    value_inline20__ssa_v0: pl.Tile[[32, 32], pl.FP32, pl.TileView(valid_shape=[count_inline18__ssa_v0, 32])] = pl.tile.load(
                        tmp_inline26__phi_v5, [row_inline24__ssa_v0, 0], [32, 32], [count_inline18__ssa_v0, 32]
                    )
                    pl.tile.store(value_inline20__ssa_v0, [row_inline24__ssa_v0, 0], out__ssa_v0)
        return out__ssa_v0
