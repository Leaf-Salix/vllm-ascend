# SPDX-License-Identifier: Apache-2.0
"""官方#2936同型CPU复现：内部分配尺寸穿过inline和循环，不运行设备。"""

import pypto.language as pl
from pypto.pypto_core import passes

ROWS = pl.dynamic("CSA_INTERNAL_CALLEE_ROWS")


@pl.program
class InternalExtent:
    @pl.function(type=pl.FunctionType.Inline)
    def fill(self, work: pl.Tensor[[ROWS, 4], pl.FP32], src: pl.Tensor[[4, 4], pl.FP32]):
        for _ in pl.range(2):
            updated = pl.tensor.assemble(work, src, [0, 0])
            work = updated
        return work

    @pl.function
    def main(self, n: pl.Scalar[pl.INDEX], src: pl.Tensor[[4, 4], pl.FP32]):
        extent = n + 4
        work = pl.create_tensor([extent, 4], dtype=pl.FP32)
        result = self.fill(work, src)
        return result


if __name__ == "__main__":
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
        program = passes.inline_functions()(InternalExtent)
        passes.convert_to_ssa()(program)
    print("PASS: dynamic internal extent survives inline and SSA verification; no device execution")
