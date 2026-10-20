# SPDX-License-Identifier: Apache-2.0
"""仅用既有load/set_validshape/add/store复现嵌套read的类型元数据自由变量。

没有新compact算子，没有NPU执行。named对照将有效长度命名为SSA标量，
数学表达、动态有效长度和输出完全相同。
"""
import argparse
import json
from pathlib import Path
import traceback

import pypto
import pypto.language as pl


@pl.jit.incore
def calculate_nested(
    src: pl.Tensor[[1, 2048], pl.FP32],
    valid: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[1, 2048], pl.FP32]],
):
    tile = pl.load(src, [0, 0], [1, 2048])
    narrowed = pl.set_validshape(tile, 1, pl.cast(pl.read(valid, [0]), pl.INDEX))
    result = pl.add(narrowed, 1.0)
    pl.store(result, [0, 0], out)
    return out


@pl.jit
def entry_nested(
    src: pl.Tensor[[1, 2048], pl.FP32],
    valid: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[1, 2048], pl.FP32]],
):
    with pl.spmd(1):
        out = calculate_nested(src, valid, out)
    return out


@pl.jit.incore
def calculate_named(
    src: pl.Tensor[[1, 2048], pl.FP32],
    valid: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[1, 2048], pl.FP32]],
):
    tile = pl.load(src, [0, 0], [1, 2048])
    n = pl.read(valid, [0])
    n_index = pl.cast(n, pl.INDEX)
    narrowed = pl.set_validshape(tile, 1, n_index)
    result = pl.add(narrowed, 1.0)
    pl.store(result, [0, 0], out)
    return out


@pl.jit
def entry_named(
    src: pl.Tensor[[1, 2048], pl.FP32],
    valid: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[1, 2048], pl.FP32]],
):
    with pl.spmd(1):
        out = calculate_named(src, valid, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    from pypto.runtime import CompileOptions

    report = dict(device_run=False, new_compact_op_used=False, cases=[])
    for named, entry in ((False, entry_nested), (True, entry_named)):
        folder = args.output / ("named" if named else "nested")
        config = CompileOptions(platform="a2a3", output_dir=str(folder), dump_passes=True)
        case = dict(named=named, passed=False)
        try:
            artifact = entry._resolve_kernel_artifact(
                (), {"config": config}, allow_signature_mode=True,
            )
            case.update(passed=True, artifact=str(artifact.output_dir))
        except BaseException as exc:
            case.update(error=str(exc), traceback=traceback.format_exc())
        for dump in sorted((folder / "passes_dump").glob("*.py")):
            if "__FREE_VAR" in dump.read_text():
                case["first_free_variable_dump"] = str(dump)
                break
        report["cases"].append(case)
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps([
        {k: v for k, v in case.items() if k not in ("error", "traceback")}
        for case in report["cases"]
    ], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
