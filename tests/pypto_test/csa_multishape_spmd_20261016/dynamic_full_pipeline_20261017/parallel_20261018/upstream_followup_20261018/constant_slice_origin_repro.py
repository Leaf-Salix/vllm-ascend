import pytest
import pypto.language as pl
from pypto import ir, passes

def test_nested_constant_origin_folds_without_freezing_loop_metadata():
    """Constant SSA origins and loop-carried origins need different handling."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pool = pl.tile.create([128, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            row: pl.Scalar[pl.INDEX] = 16
            for iteration in pl.range(2):
                base: pl.Scalar[pl.INDEX] = 32
                fixed_row = base * 2
                constant_view = pl.tile.slice(pool, [16, 16], [fixed_row, 0])
                carried_view = pl.tile.slice(pool, [16, 16], [row, 0])
                indexed_view = pl.tile.slice(pool, [16, 16], [iteration * 16, 0])
                _first = pl.tile.store(constant_view, [0, 0], output)
                _second = pl.tile.store(carried_view, [0, 0], output)
                _third = pl.tile.store(indexed_view, [0, 0], output)
                row = row + 16
            return output

    after = passes.init_mem_ref()(passes.convert_to_ssa()(Before))
    found = {}

    class Origins(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):
            for name in ("constant_view", "carried_view", "indexed_view"):
                if stmt.var.name_hint.startswith(name):
                    found[name] = stmt.var.type.memref.byte_offset_
            super().visit_assign_stmt(stmt)

    Origins().visit_stmt(next(iter(after.functions.values())).body)
    assert set(found) == {"constant_view", "carried_view", "indexed_view"}
    assert isinstance(found["constant_view"], ir.ConstInt)
    assert found["constant_view"].value == 64 * 16 * 4
    assert not isinstance(found["carried_view"], ir.ConstInt)
    assert not isinstance(found["indexed_view"], ir.ConstInt)

if __name__ == "__main__":
    pytest.main([__file__, "-q"])
