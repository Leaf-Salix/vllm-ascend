# SPDX-License-Identifier: Apache-2.0
import pypto.language as pl
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.pypto_core import passes

backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

def _collect_named_tile_memrefs(program: ir.Program, names: set[str]) -> dict[str, ir.MemRef]:
    """Collect MemRefs for named tile definitions anywhere in a function body."""
    result: dict[str, ir.MemRef] = {}
    function = next(iter(program.functions.values()))

    class _MemRefCollector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):  # type: ignore[override]
            tile_type = stmt.var.type
            if stmt.var.name_hint in names and isinstance(tile_type, ir.TileType):
                assert tile_type.memref is not None
                result[stmt.var.name_hint] = tile_type.memref
            super().visit_assign_stmt(stmt)

    _MemRefCollector().visit_stmt(function.body)
    missing = names - result.keys()
    assert not missing, f"tile definitions not found: {sorted(missing)}"
    return result


@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        pool: pl.Tensor[[16, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[8, 64], pl.FP16]],
    ) -> pl.Tensor[[8, 64], pl.FP16]:
        current: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
        stale: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
        seed: pl.Tile[[16, 32], pl.FP16, pl.MemRef(current, 0, 1024), pl.Mem.Mat] = (
            pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        )
        gathered: pl.Tile[[16, 32], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.gather_row(seed, pool, [0, 0], [0, 0], [16, 32])
        )
        transposed: pl.Tile[[32, 16], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.transpose_view(gathered)
        )
        reshaped: pl.Tile[[8, 64], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.reshape(transposed, [8, 64])
        )
        return pl.tile.store(reshaped, [0, 0], out)

after = passes.materialize_semantic_aliases()(Before)
refs = _collect_named_tile_memrefs(after, {"seed", "gathered", "transposed", "reshaped"})
for name in ("gathered", "transposed", "reshaped"):
    assert ir.MemRef.same_allocation(refs[name], refs["seed"])
    assert ir.structural_equal(refs[name].byte_offset_, refs["seed"].byte_offset_)
    assert refs[name].size_ == 1024
# A second fixed pass is a no-op; this must not need a fixed-point loop.
ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)

