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
        waves: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ) -> pl.Tensor[[16, 32], pl.FP16]:
        seed = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        for (wave,) in pl.while_(init_values=(0,)):
            pl.cond(wave < waves)
            for i, (carry,) in pl.range(2, init_values=(seed,)):
                replacement = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
                result = pl.yield_(replacement)
            _read_result = pl.tile.extract(result, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            _wave_out = pl.yield_(wave + 1)
        return out

refs = _collect_named_tile_memrefs(passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before)), {"seed", "replacement"})
assert ir.MemRef.same_allocation(refs["seed"], refs["replacement"])

