# pypto.program: Repro
import pypto.language as pl


@pl.program
class Repro:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def core(x: pl.Tensor[[4, 512], pl.BF16], indices: pl.Tensor[[1, 16], pl.INT32], out: pl.Tensor[[2, 512], pl.BF16]) -> pl.Tensor[[2, 512], pl.BF16]:
        ids: pl.Tile[[1, 16], pl.INT32] = pl.tile.load(indices, [0, 0], [1, 16], [1, 16])
        pitch: pl.Scalar[pl.INDEX] = pl.cast(pl.tile.read(ids, [0, 0]), pl.INDEX)
        if pitch == 1024:
            alias: pl.Tensor[[2, pitch], pl.BF16, pl.TensorView(stride=[pitch, 1], layout=pl.TensorLayout.ND)] = pl.tensor.view(x, [2, pitch])
            value: pl.Tile[[2, 512], pl.BF16] = pl.tile.load(alias, [0, 0], [2, 512], [2, 512])
            out: pl.Tensor[[2, 512], pl.BF16] = pl.tile.store(value, [0, 0], out)
        return out