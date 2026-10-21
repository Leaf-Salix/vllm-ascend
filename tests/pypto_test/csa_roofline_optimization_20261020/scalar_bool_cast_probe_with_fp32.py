# SPDX-License-Identifier: Apache-2.0
"""独立核对Scalar BOOL转数值；区分literal、constexpr和device比较结果。"""

import argparse
import json
import os
from pathlib import Path
import traceback

import pypto
import pypto.language as pl


@pl.jit.incore
def casts(src: pl.Tensor[[1], pl.INT32], ints: pl.InOut[pl.Tensor[[8], pl.INT32]],
          floats: pl.InOut[pl.Tensor[[4], pl.FP32]], ENABLED: pl.constexpr):
    pl.system.cacheinvalid()
    pl.system.fence()
    value = pl.read(src, [0])
    condition = value > 0
    literal_index = pl.cast(True, pl.INDEX)
    runtime_index = pl.cast(condition, pl.INDEX)
    pl.write(ints, [0], pl.cast(True, pl.INT32))
    pl.write(ints, [1], pl.cast(False, pl.INT32))
    pl.write(ints, [2], pl.cast(ENABLED, pl.INT32))
    pl.write(ints, [3], pl.cast(condition, pl.INT32))
    pl.write(ints, [4], pl.cast(literal_index, pl.INT32))
    pl.write(ints, [5], pl.cast(runtime_index, pl.INT32))
    pl.write(floats, [0], pl.cast(True, pl.FP32))
    pl.write(floats, [1], pl.cast(condition, pl.FP32))
    pl.system.bar_all()
    pl.system.cacheinvalid()
    pl.system.fence()
    return ints, floats


@pl.jit
def entry(src: pl.Tensor[[1], pl.INT32], ints: pl.InOut[pl.Tensor[[8], pl.INT32]],
          floats: pl.InOut[pl.Tensor[[4], pl.FP32]], ENABLED: pl.constexpr):
    with pl.spmd(1):
        ints, floats = casts(src, ints, floats, ENABLED)
    return ints, floats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, numerical_cast_matches_python=False, cases=[],
                  int_columns=["literal_true", "literal_false", "constexpr_true", "runtime_bool", "literal_via_index", "runtime_via_index"])
    try:
        import torch
        import torch_npu  # noqa: F401
        from pypto import torch as pypto_torch
        from pypto.runtime import CompileOptions
        from pypto import passes
        device = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == device
        torch.npu.set_device(device)
        pypto_torch.init(device=device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        pypto.configure_cache(pypto.CacheConfig(enabled=False))
        config = CompileOptions(platform="a2a3", output_dir=str(args.output / "compiled"),
                                memory_planner=passes.MemoryPlanner.PYPTO)
        src = torch.ones(1, dtype=torch.int32, device=f"npu:{device}")
        ints = torch.full((8,), 777, dtype=torch.int32, device=src.device)
        floats = torch.full((4,), 777.0, dtype=torch.float32, device=src.device)
        for value in (1, 0):
            src.fill_(value)
            entry(src, ints, floats, ENABLED=True, config=config)
            torch.npu.synchronize()
            actual_i, actual_f = ints.cpu().tolist(), floats.cpu().tolist()
            expected_i = [1, 0, 1, int(value > 0), 1, int(value > 0), 777, 777]
            expected_f = [1.0, float(value > 0), 777.0, 777.0]
            report["cases"].append(dict(input=value, actual_int=actual_i, actual_float=actual_f,
                                        expected_int=expected_i, expected_float=expected_f,
                                        matches_python=actual_i == expected_i and actual_f == expected_f))
        report.update(completed=True, numerical_cast_matches_python=all(row["matches_python"] for row in report["cases"]))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
