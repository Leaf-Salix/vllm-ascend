# pypto.program: _jit_entry
import pypto.language as pl


@pl.program
class _jit_entry:
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def entry_incore_0_aic(
        x__ssa_v0: pl.Tensor[[32, 128], pl.BF16], weight__ssa_v0: pl.Tensor[[128, 32], pl.BF16], out__ssa_v0: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks__ssa_v0: pl.Out[pl.Tensor[[2, 16], pl.INT32]]
    ):
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for query__idx_v0, (previous, recent) in pl.range(worker__ssa_v0, 2, init_values=(0, 0)):
            for step__idx_v0, (old, new) in pl.range(3, init_values=(previous, recent)):
                source__ssa_v0: pl.Scalar[pl.INDEX] = step__idx_v0 % 2 * 16
                if step__idx_v0 < 2:
                    saved__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(source__ssa_v0)
                else:
                    saved__rv_v0: pl.Scalar[pl.INDEX] = pl.yield_(new)
                if 1 <= step__idx_v0:
                    for chunk__idx_v0 in pl.pipeline(2, stage=2):
                        lhs__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(x__ssa_v0, [old, chunk__idx_v0 * 64], [16, 64], [16, 64], target_memory=pl.Mem.Mat)
                        rhs__ssa_v0: pl.Tile[[64, 32], pl.BF16, pl.Mem.Mat] = pl.tile.load(weight__ssa_v0, [chunk__idx_v0 * 64, 0], [64, 32], [64, 32], target_memory=pl.Mem.Mat)
                        lhs__ssa_v0_Left: pl.Tile[[16, 64], pl.BF16, pl.Mem.Left] = pl.tile.move(lhs__ssa_v0, target_memory=pl.Mem.Left)
                        rhs__ssa_v0_Right: pl.Tile[[64, 32], pl.BF16, pl.Mem.Right] = pl.tile.move(rhs__ssa_v0, target_memory=pl.Mem.Right)
                        product__ssa_v0: pl.Tile[[16, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs__ssa_v0_Left, rhs__ssa_v0_Right)
                        out__store: pl.Tensor[[16, 32], pl.FP32] = pl.tile.store(product__ssa_v0, [0, 0], out__ssa_v0)
                old_done, new_done = pl.yield_(new, saved__rv_v0)
            previous_done, recent_done = pl.yield_(old_done, new_done)

    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def entry_incore_0_aiv(
        x__ssa_v0: pl.Tensor[[32, 128], pl.BF16], weight__ssa_v0: pl.Tensor[[128, 32], pl.BF16], out__ssa_v0: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks__ssa_v0: pl.Out[pl.Tensor[[2, 16], pl.INT32]]
    ) -> tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[2, 16], pl.INT32]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        lane__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        marker__ssa_v0: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
        marks__store: pl.Tensor[[2, 16], pl.INT32] = pl.tile.store(marker__ssa_v0, [lane__ssa_v0, 0], marks__ssa_v0)
        return out__ssa_v0, marks__ssa_v0

    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def entry_incore_0(
        x__ssa_v0: pl.Tensor[[32, 128], pl.BF16], weight__ssa_v0: pl.Tensor[[128, 32], pl.BF16], out__ssa_v0: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks__ssa_v0: pl.Out[pl.Tensor[[2, 16], pl.INT32]]
    ) -> tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[2, 16], pl.INT32]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        self.entry_incore_0_aic(x__ssa_v0, weight__ssa_v0, out__ssa_v0, marks__ssa_v0)
        self.entry_incore_0_aiv(x__ssa_v0, weight__ssa_v0, out__ssa_v0, marks__ssa_v0)
        return out__ssa_v0, marks__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def entry_spmd_0(
        self,
        x__ssa_v0: pl.Tensor[[32, 128], pl.BF16],
        weight__ssa_v0: pl.Tensor[[128, 32], pl.BF16],
        out__ssa_v0: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        marks__ssa_v0: pl.Out[pl.Tensor[[2, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[2, 16], pl.INT32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[2, 16], pl.INT32]] = self.entry_incore_0(x__ssa_v0, weight__ssa_v0, out__ssa_v0, marks__ssa_v0)
        out__ssa_v1: pl.Tensor[[16, 32], pl.FP32] = ret__tmp_v0[0]
        marks__ssa_v1: pl.Tensor[[2, 16], pl.INT32] = ret__tmp_v0[1]
        return out__ssa_v0, marks__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def entry(
        self,
        x__ssa_v0: pl.Tensor[[32, 128], pl.BF16],
        weight__ssa_v0: pl.Tensor[[128, 32], pl.BF16],
        out__ssa_v0: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        marks__ssa_v0: pl.Out[pl.Tensor[[2, 16], pl.INT32]],
    ) -> tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[2, 16], pl.INT32]]:
        pl.func_attr({"mx_tensor_views_blocked": True})
        ret__tmp_v0: pl.Tuple[pl.Tensor[[16, 32], pl.FP32], pl.Tensor[[2, 16], pl.INT32]] = self.entry_spmd_0(x__ssa_v0, weight__ssa_v0, out__ssa_v0, marks__ssa_v0, attrs={"core_num": 1})
        out__ssa_v1: pl.Tensor[[16, 32], pl.FP32] = ret__tmp_v0[0]
        marks__ssa_v1: pl.Tensor[[2, 16], pl.INT32] = ret__tmp_v0[1]
        return out__ssa_v1, marks__ssa_v1
