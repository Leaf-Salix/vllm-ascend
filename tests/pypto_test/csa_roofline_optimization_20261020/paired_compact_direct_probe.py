# SPDX-License-Identifier: Apache-2.0
"""直接核对成对压缩的原始输出，防止完整排序回退掩盖错误。

覆盖GE/GT/EQ、动态有效长度、零长度、容量截断、超FP32精确范围的INT32基址、
FP32位保留及同地址变输入图重放。只证明原语，不代替完整CSA精度和性能。
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import pypto
import pypto.language as pl

sys.path.insert(0, str(Path(__file__).resolve().parent / "source_topk_paired"))
from topk_compact_probe import fixtures

ROWS = pl.dynamic("PAIRED_COMPACT_DIRECT_ROWS")


@pl.jit.incore
def compact_rows(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    valid: pl.Tensor[[ROWS], pl.INT32],
    thresholds: pl.Tensor[[ROWS], pl.FP32],
    offsets: pl.Tensor[[ROWS], pl.INT32],
    values_out: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
    ids_out: pl.InOut[pl.Tensor[[ROWS, 1040], pl.INT32]],
    counts_out: pl.InOut[pl.Tensor[[ROWS, 16], pl.INT32]],
    MODE: pl.constexpr,
):
    for row in pl.range(pl.tensor.dim(src, 0)):
        values = pl.load(src, [row, 0], [1, 2048])
        values = pl.set_validshape(values, 1, pl.cast(pl.read(valid, [row]), pl.INDEX))
        threshold = pl.read(thresholds, [row])
        offset = pl.read(offsets, [row])
        scratch = pl.tile.create([1, 18432], dtype=pl.UINT8)
        packed, ids, count = pl.tile.compact_compare(
            values, threshold, scratch, offset, cmp_mode=MODE, out_cols=1024,
        )
        pl.store(packed, [row, 8], values_out)
        pl.store(ids, [row, 8], ids_out)
        counts_out = pl.write(counts_out, [row, 8], pl.tile.read(count, [0, 0]))
    return values_out, ids_out, counts_out


@pl.jit
def entry(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    valid: pl.Tensor[[ROWS], pl.INT32],
    thresholds: pl.Tensor[[ROWS], pl.FP32],
    offsets: pl.Tensor[[ROWS], pl.INT32],
    values_out: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
    ids_out: pl.InOut[pl.Tensor[[ROWS, 1040], pl.INT32]],
    counts_out: pl.InOut[pl.Tensor[[ROWS, 16], pl.INT32]],
    MODE: pl.constexpr,
):
    with pl.spmd(1):
        values_out, ids_out, counts_out = compact_rows(
            src, valid, thresholds, offsets, values_out, ids_out, counts_out, MODE,
        )
    return values_out, ids_out, counts_out


def make_inputs(torch, seed):
    names, src, old, valid = fixtures(torch, seed)
    thresholds = old[:, 1022].contiguous()
    extra = torch.zeros((6, 2048), dtype=torch.float32)
    extra[:, 0] = -0.0
    extra[:, 1] = float("inf")
    extra[:, 2] = float("-inf")
    extra[:, 3] = float("nan")
    lengths = [0, 1, 63, 64, 65, 2048]
    src = torch.cat([src, extra])
    valid = torch.cat([valid, torch.tensor(lengths, dtype=torch.int32)])
    thresholds = torch.cat([thresholds, torch.zeros(6)])
    names += [{"valid": n, "bit_pattern_case": True} for n in lengths]
    # 这些整数超出FP32连续精确表示范围，能暴露把索引经浮点转换的错误。
    bases = [17, 16777217, -16777217]
    offsets = torch.tensor([bases[(i + seed) % 3] for i in range(len(names))],
                           dtype=torch.int32)
    return names, (src, valid, thresholds, offsets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    report = {"passed": False, "full_csa_validated": False, "cases": []}
    try:
        from pypto import torch as pypto_torch
        import torch
        import torch_npu
        from pypto import passes
        from pypto.runtime import CompileOptions

        dev = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == dev
        torch.npu.set_device(dev)
        pypto_torch.init(device=dev, platform="a2a3", runtime="tensormap_and_ringbuffer")
        names, host = make_inputs(torch, 1024)
        inputs = [x.to(f"npu:{dev}") for x in host]
        values = torch.full((len(names), 1040), -777.0, device=inputs[0].device)
        ids = torch.full(values.shape, -777, dtype=torch.int32, device=values.device)
        counts = torch.full((len(names), 16), -777, dtype=torch.int32, device=values.device)
        for mode, compare in (("ge", torch.ge), ("gt", torch.gt), ("eq", torch.eq)):
            config = CompileOptions(
                platform="a2a3", output_dir=str(args.output / mode),
                dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO,
            )
            entry(*inputs, values, ids, counts, mode, config=config)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                entry(*inputs, values, ids, counts, mode, config=config)
            for seed in (1024, 2048, 1024):
                names, host = make_inputs(torch, seed)
                for device_tensor, cpu_tensor in zip(inputs, host, strict=True):
                    device_tensor.copy_(cpu_tensor)
                values.fill_(-777)
                ids.fill_(-777)
                counts.fill_(-777)
                graph.replay()
                torch.npu.synchronize()
                actual_v, actual_i, actual_c = values.cpu(), ids.cpu(), counts.cpu()
                failures = []
                for row, name in enumerate(names):
                    n = int(host[1][row])
                    selected = compare(host[0][row, :n], host[2][row]).nonzero().flatten()
                    kept = len(selected)
                    selected = selected[:1024]
                    expected_v = host[0][row, selected]
                    expected_i = selected.int() + host[3][row]
                    good = int(actual_c[row, 8]) == kept
                    good &= torch.equal(actual_v[row, 8:8 + len(selected)].view(torch.int32),
                                        expected_v.view(torch.int32))
                    good &= torch.equal(actual_i[row, 8:8 + len(selected)], expected_i)
                    if not good:
                        failures.append({"row": row, **name, "expected_count": kept,
                                         "actual_count": int(actual_c[row, 8])})
                guards = all(bool((x[:, :8] == -777).all() and (x[:, -8:] == -777).all())
                             for x in (actual_v, actual_i))
                guards &= bool((actual_c[:, :8] == -777).all() and
                               (actual_c[:, 9:] == -777).all())
                inputs_ok = all(torch.equal(x.cpu().view(torch.int32), y.view(torch.int32))
                                for x, y in zip(inputs, host, strict=True))
                report["cases"].append({
                    "mode": mode, "seed": seed, "rows": len(names), "failures": failures,
                    "guards": guards, "inputs_unchanged": inputs_ok,
                })
                assert not failures and guards and inputs_ok, report["cases"][-1]
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False),
          flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
