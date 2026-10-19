"""独立 CPU 复现：合法 NZ 全量展平跨 JIT helper 的元数据丢失。"""
import argparse
import json
from pathlib import Path

import pypto
import pypto.language as pl


@pl.jit.incore
def consume(w: pl.Tensor[[131072], pl.INT8], out: pl.Out[pl.Tensor[[16], pl.INT8]]):
    value = pl.load(w, [0], [16])
    out = pl.store(value, [0], out)
    return out


@pl.jit
def entry(w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
          out: pl.Out[pl.Tensor[[16], pl.INT8]]):
    flat = pl.reshape(w, [131072])
    with pl.spmd(1):
        out = consume(flat, out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"pypto": pypto.__file__, "device_execution": False}
    try:
        entry.specialize()
        result["specialization_passed"] = True
    except Exception as exc:
        result.update(specialization_passed=False, error_type=type(exc).__name__, error=str(exc))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
