# SPDX-License-Identifier: Apache-2.0
"""只验证Score WS的Cube/Vector归约语义，不作为CSA或带宽性能成绩。

WS稠密对角系数包含大量零项。若按小panel交接给Vector，容量问题可通过分段
处理，但交接总字节仍增大，且FP32归约顺序必须先核实。本探针只回答数值
等价这一必要条件：数学tile为6 query、64 head、64候选，不固化CSA的B/T/L。
输入是独立算术探针的有限FP16值，不是模型权重测试；正式整层仍用W8A8层4。
"""

import argparse
import json
import os
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def cube_ws(coeff: pl.Tensor[[16, 384], pl.FP16], qk: pl.Tensor[[384, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]):
    left = pl.load(coeff, [0, 0], [16, 384], target_memory=pl.MemorySpace.Mat)
    right = pl.load(qk, [0, 0], [384, 64], target_memory=pl.MemorySpace.Mat)
    summed = pl.matmul(left, right, out_dtype=pl.FP32)
    pl.store(summed, [0, 0], out)
    return out


@pl.jit.incore
def vector_ws(coeff: pl.Tensor[[16, 384], pl.FP16], qk: pl.Tensor[[384, 64], pl.FP16],
              out: pl.Out[pl.Tensor[[6, 64], pl.FP32]]):
    for query in pl.range(6):
        panel = pl.load(qk, [query * 64, 0], [64, 64])
        weights = pl.load(coeff, [query, query * 64], [1, 64])
        # 两输入先升FP32，避免在乘积处增加一次FP16舍入；仍须实测归约树。
        transposed = pl.transpose(panel, axis1=0, axis2=1)
        products = pl.col_expand_mul(pl.cast(transposed, pl.FP32), pl.cast(weights, pl.FP32))
        reduced = pl.row_sum(products, tmp_tile=pl.create_tile([64, 64], dtype=pl.FP32))
        pl.store(pl.reshape(reduced, [1, 64]), [query, 0], out)
    return out


@pl.jit
def cube_entry(coeff: pl.Tensor[[16, 384], pl.FP16], qk: pl.Tensor[[384, 64], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]):
    with pl.spmd(1):
        out = cube_ws(coeff, qk, out)
    return out


@pl.jit
def vector_entry(coeff: pl.Tensor[[16, 384], pl.FP16], qk: pl.Tensor[[384, 64], pl.FP16],
                 out: pl.Out[pl.Tensor[[6, 64], pl.FP32]]):
    with pl.spmd(1):
        out = vector_ws(coeff, qk, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pypto
    import pypto.torch
    import torch
    import torch_npu  # noqa: F401
    from pypto import passes, pypto_core
    from pypto.runtime import CompileOptions

    device_id = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device_id)
    pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
    device = torch.device(f"npu:{device_id}")
    generator = torch.Generator().manual_seed(20261018)
    config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
    report = dict(scope="独立WS归约语义", pypto=pypto.__file__, core=pypto_core.__file__,
                  csa_validated=False, performance_measured=False, cases=[])
    for name in ("exact_binary_control", "positive_half", "signed_coefficients", "wide_exponents"):
        if name == "exact_binary_control":
            values = (torch.arange(384 * 64).reshape(384, 64) % 8).float() / 8
            weights = torch.full((6, 64), 1 / 64)
        else:
            values = torch.randn((384, 64), generator=generator).relu()
            weights = torch.randn((6, 64), generator=generator) * 0.125
            if name == "positive_half":
                values *= 1 / 1024
                weights = weights.abs()
            elif name == "wide_exponents":
                values *= torch.pow(2.0, torch.randint(-8, 8, (384, 64), generator=generator))
        values = values.half()
        weights = weights.half()
        coeff = torch.zeros((16, 384), dtype=torch.float16)
        for query in range(6):
            coeff[query, query * 64:(query + 1) * 64] = weights[query]
        coeff_npu, values_npu = coeff.to(device), values.to(device)
        cube_out = torch.full((16, 64), -777.0, dtype=torch.float32, device=device)
        vector_out = torch.full((6, 64), -777.0, dtype=torch.float32, device=device)
        cube_entry(coeff_npu, values_npu, cube_out, config=config)
        vector_entry(coeff_npu, values_npu, vector_out, config=config)
        torch.npu.synchronize()
        cube, vector = cube_out.cpu()[:6], vector_out.cpu()
        exact_sum = (coeff.double() @ values.double())[:6].float()
        different = cube.view(torch.int32) != vector.view(torch.int32)
        count = int(different.sum())
        row = dict(name=name, elements=cube.numel(), bit_differences=count,
                   max_abs=float((cube - vector).abs().max()),
                   cube_vs_rounded_fp64=int((cube != exact_sum).sum()),
                   vector_vs_rounded_fp64=int((vector != exact_sum).sum()))
        snapshot = args.output / (name + ".pt")
        torch.save(dict(coeff=coeff, qk=values, cube=cube, vector=vector), snapshot)
        row["snapshot"] = str(snapshot.resolve())
        row["snapshot_bytes"] = snapshot.stat().st_size
        report["cases"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if name == "exact_binary_control":
            assert count == 0 and torch.equal(cube, exact_sum), "先检查无舍入对照的读取/布局"
    report["all_bit_equal"] = all(row["bit_differences"] == 0 for row in report["cases"])
    report["probe_completed"] = True
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
