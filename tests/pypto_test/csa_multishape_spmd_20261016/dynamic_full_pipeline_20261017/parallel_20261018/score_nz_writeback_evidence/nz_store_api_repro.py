# SPDX-License-Identifier: Apache-2.0
"""复现现有NZ只读边界：AIC FP32 Acc不能写NZ GM；只运行编译pass。"""

import argparse
import json
from pathlib import Path

import pypto
import pypto.language as pl
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.pypto_core import passes


@pl.program
class StoreAccumulatorToNz:
    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        x: pl.Tensor[[64, 64], pl.FP16],
        w: pl.Tensor[[64, 16], pl.FP16],
        out: pl.Tensor[[64, 16], pl.FP32, pl.NZ],
    ):
        lhs = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        rhs = pl.load(w, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        accumulator = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
        pl.store(accumulator, [0, 0], out)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    prefix = (
        "inline_functions", "unroll_loops", "ctrl_flow_transform", "convert_to_ssa",
        "simplify", "normalize_stmt_structure", "flatten_call_expr", "outline_hierarchy_scopes",
        "outline_incore_scopes", "outline_cluster_scopes", "convert_tensor_to_tile_ops",
        "optimize_orch_tensors", "lower_composite_ops", "flatten_tile_nd_to_2d", "block_nz_tensor_views",
    )
    program = StoreAccumulatorToNz
    report = dict(api_supported=False, restriction_reproduced=False, device_execution=False,
                  pypto=str(pypto.__file__), completed_passes=[])
    try:
        for name in prefix:
            report["current_pass"] = name
            program = getattr(passes, name)()(program)
            report["completed_passes"].append(name)
        report["api_supported"] = True
    except ValueError as error:
        report["error"] = str(error)
        report["restriction_reproduced"] = "NZ layout is read-only" in str(error)
        if not report["restriction_reproduced"]:
            raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
    assert report["restriction_reproduced"], "当前限制已变化，需重新核对，不能按旧结论提issue"


if __name__ == "__main__":
    main()
