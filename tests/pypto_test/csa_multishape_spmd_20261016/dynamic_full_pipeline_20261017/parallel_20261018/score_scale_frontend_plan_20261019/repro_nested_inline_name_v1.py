# SPDX-License-Identifier: Apache-2.0
"""最小 CPU 复现：嵌套 inline 表达式的 hoisted 临时名再内联后无法进入 SSA。"""
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print


@pl.program
class Nested:
    @pl.function(type=pl.FunctionType.Inline)
    def base(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        group_base = (requests + 16) // 16 * 16
        group_end = group_base + 4 * tokens
        return (group_end + 15) // 16 * 16

    @pl.function(type=pl.FunctionType.Inline)
    def layout(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        return self.base(requests, tokens) + 48 + 4 * tokens

    @pl.function
    def main(self, requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        result = self.layout(requests, tokens)
        return result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    report = dict(inline=False, ssa=False, device_execution=False)
    try:
        inline = passes.inline_functions()(Nested)
        report["inline"] = True
        (here / "nested_inline_after.py").write_text(python_print(inline))
        passes.convert_to_ssa()(inline)
        report["ssa"] = True
    except Exception as exc:
        report["exception"] = type(exc).__name__
        report["message"] = str(exc)
    (here / "NESTED_INLINE_REPRO.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
