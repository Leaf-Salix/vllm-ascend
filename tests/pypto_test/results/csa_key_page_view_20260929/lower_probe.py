"""CPU-only lowering probe for a Native page byte slice followed by a key matrix view."""
import json
from pathlib import Path

import pypto.language as pl
from pypto import backend, codegen, ir
from pypto.backend import BackendType

ROOT = Path(__file__).resolve().parent


@pl.program
class NativePageView:
    @pl.function(type=pl.FunctionType.InCore)
    def page_qk(
        self,
        cache: pl.Tensor[[16, 4160], pl.INT8],
        query: pl.Tensor[[16, 128], pl.INT8],
        page: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[16, 32], pl.INT32]],
    ) -> pl.Tensor[[16, 32], pl.INT32]:
        physical = pl.min(pl.max(page, 0), 15)
        key_bytes = pl.tensor.slice(cache, [1, 4096], [physical, 0])
        key_matrix = pl.reshape(key_bytes, [32, 128])
        key_l1 = pl.load(key_matrix, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat)
        query_l1 = pl.load(query, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        query_l0 = pl.tile.move(query_l1, target_memory=pl.MemorySpace.Left)
        key_l0 = pl.tile.move(pl.tile.transpose_view(key_l1), target_memory=pl.MemorySpace.Right)
        dot = pl.tile.matmul(query_l0, key_l0)
        return pl.store(dot, [0, 0], out)


def main():
    backend.set_backend_type(BackendType.Ascend910B)
    program = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(NativePageView)
    value = codegen.PTOCodegen().generate(program)
    mlir = value if isinstance(value, str) else "\n".join(value.values())
    (ROOT / "page_view.mlir").write_text(mlir)
    report = {"status": "LOWERING_PASS", "scope": "PyPTO pipeline and PTO MLIR only; no CCE or device",
              "native_page_bytes": 4160, "key_shape": [32, 128], "source": "PyPTO88f60598"}
    (ROOT / "lowering.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
