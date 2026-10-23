# SPDX-License-Identifier: Apache-2.0
"""CPU-only PyPTO reproduction: a known-false guard must protect its division."""

import json

import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print


@pl.program
class Repro:
    @pl.function(type=pl.FunctionType.Inline)
    def lookup(self, query: pl.Scalar[pl.INDEX], width: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        owner = pl.cast(-1, pl.INDEX)
        if width > 0:
            owner = query // width
        else:
            owner = pl.cast(0, pl.INDEX)
        return owner

    @pl.function
    def main(self, query: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        width = pl.cast(0, pl.INDEX)
        owner = self.lookup(query, width)
        return owner


@pl.program
class SafeDivisor:
    @pl.function(type=pl.FunctionType.Inline)
    def lookup(self, query: pl.Scalar[pl.INDEX], width: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        owner = pl.cast(-1, pl.INDEX)
        if width > 0:
            owner = query // pl.max(width, 1)
        else:
            owner = pl.cast(0, pl.INDEX)
        return owner

    @pl.function
    def main(self, query: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        width = pl.cast(0, pl.INDEX)
        owner = self.lookup(query, width)
        return owner


def main():
    report = {"device_execution": False, "cases": {}}
    for name, program in [("guarded_zero", Repro), ("safe_divisor", SafeDivisor)]:
        stage = "InlineFunctions"
        try:
            inlined = passes.inline_functions()(program)
            stage = "Simplify"
            simplified = passes.simplify()(inlined)
            report["cases"][name] = {"passed": True, "simplified": python_print(simplified)}
        except Exception as exc:
            report["cases"][name] = {"passed": False, "stage": stage, "error": str(exc)}
    print(json.dumps(report, indent=2))
    assert not report["cases"]["guarded_zero"]["passed"]
    assert "Floor division by zero" in report["cases"]["guarded_zero"]["error"]
    assert report["cases"]["safe_divisor"]["passed"]


if __name__ == "__main__":
    main()
