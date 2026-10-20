# SPDX-License-Identifier: Apache-2.0
"""验证 A3 排序同分顺序能否在保序阈值压缩后保持；不是完整性能测试。

参考输入按原 CSA 的 2048 排序树执行，候选输入由 CPU 预先保序筛选并填
最小值，设备按保留项数量走 512/1024/2048 的原排序树。CPU 预筛选仅用于
隔离排序语义；正式算子必须在 AIV 上计算掩码和计数，不引入 Host 输入。
覆盖随机分数、密集同分、跨 32/128/512 边界、零候选和阈值相等项。
输出比较包含 index 位，不能仅比较 score 集合。
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import pypto.language as pl

ROWS = pl.dynamic("TOPK_TIE_PROBE_ROWS")
NEG = -3.4028234663852886e38


@pl.jit.inline(auto_scope=False)
def sort_segment(values: pl.Tile, indices: pl.Tile, count: pl.Scalar[pl.INT32]):
    if count <= 512:
        v = pl.tile.extract(values, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec)
        i = pl.tile.extract(indices, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec)
        pairs = pl.sort32(v, pl.reinterpret_view(i, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        top = pl.tile.extract(pairs, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
    elif count <= 1024:
        v = pl.tile.extract(values, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
        i = pl.tile.extract(indices, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
        pairs = pl.sort32(v, pl.reinterpret_view(i, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        left = pl.tile.slice(pairs, [1, 1024], [0, 0])
        right = pl.tile.slice(pairs, [1, 1024], [0, 1024])
        tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
        merged = pl.tile.mrgsort(left, right, tmp=tmp)
        top = pl.tile.extract(merged, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
    else:
        pairs = pl.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        pairs = pl.mrgsort(pairs, block_len=1024)
        top = pl.tile.extract(pairs, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
    return top


@pl.jit.incore
def sort_rows(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    indices: pl.Tensor[[ROWS, 2048], pl.INT32],
    counts: pl.Tensor[[ROWS], pl.INT32],
    out: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
):
    for row in pl.range(pl.tensor.dim(src, 0)):
        vals = pl.load(src, [row, 0], [1, 2048])
        idx = pl.load(indices, [row, 0], [1, 2048])
        n = pl.read(counts, [row])
        top = sort_segment(vals, idx, n)
        pl.store(top, [row, 8], out)
    return out


@pl.jit
def entry(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    indices: pl.Tensor[[ROWS, 2048], pl.INT32],
    counts: pl.Tensor[[ROWS], pl.INT32],
    out: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
):
    with pl.spmd(1):
        out = sort_rows(src, indices, counts, out)
    return out


def fixtures(torch):
    gen = torch.Generator().manual_seed(20261020)
    cases = []
    for target in (0, 1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2048):
        order = torch.randperm(2048, generator=gen)
        for kind in ("equal", "ties", "random"):
            x = torch.full((2048,), -10.0)
            n = target
            vals = torch.ones(n)
            if kind == "ties":
                vals = torch.randint(0, 8, (n,), generator=gen).float()
            elif kind == "random":
                vals = torch.rand(n, generator=gen)
            x[order[:n]] = vals
            cases.append((f"{kind}_{n}", x, torch.arange(2048, dtype=torch.int32), 0.0))
    # 检验 tie 是否看 payload index。真实 CSA index 单调，此组是诊断证据。
    cases.append(("all_equal_reverse_index", torch.ones(2048),
                  torch.arange(2047, -1, -1, dtype=torch.int32), 0.0))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--device", type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto import passes
    from pypto.runtime import CompileOptions, ensure_pto_isa_root

    config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"),
                            dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
    report = dict(passed=False, cpu_only=args.cpu_only, full_csa_validated=False,
                  performance_measured=False, cases=[])
    try:
        report["pto_isa_root"] = ensure_pto_isa_root()
        if args.cpu_only:
            artifact = entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
            report["artifact"] = str(artifact.output_dir)
        else:
            import pypto.torch
            import torch
            import torch_npu  # noqa: F401

            dev = int(os.environ["TASK_DEVICE"])
            assert args.device is None or args.device == dev
            torch.npu.set_device(dev)
            pypto.torch.init(device=dev, platform="a2a3", runtime="tensormap_and_ringbuffer")
            device = f"npu:{dev}"
            cases = fixtures(torch)
            originals, packed, original_idx, packed_idx, counts = [], [], [], [], []
            for _, x, idx, threshold in cases:
                keep = x >= threshold
                count = int(keep.sum())
                packed_x = torch.full((2048,), NEG)
                packed_i = torch.full((2048,), -1, dtype=torch.int32)
                packed_x[:count] = x[keep]
                packed_i[:count] = idx[keep]
                originals.append(x)
                original_idx.append(idx)
                packed.append(packed_x)
                packed_idx.append(packed_i)
                counts.append(count)
            src = torch.stack(originals + packed).to(device)
            idx = torch.stack(original_idx + packed_idx).to(device)
            count_tensor = torch.tensor([2048] * len(cases) + counts, dtype=torch.int32, device=device)
            out = torch.full((len(cases) * 2, 1040), -7777.0, device=device)
            entry(src, idx, count_tensor, out, config=config)
            torch.npu.synchronize()
            output = out.cpu()
            assert torch.all(output[:, :8] == -7777) and torch.all(output[:, 1032:] == -7777)
            for row, (name, x, _, _) in enumerate(cases):
                n = min(counts[row], 512)
                ref = output[row, 8:8 + 2 * n]
                got = output[row + len(cases), 8:8 + 2 * n]
                exact = torch.equal(ref.view(torch.int32), got.view(torch.int32))
                score_exact = torch.equal(ref[::2], got[::2])
                report["cases"].append(dict(name=name, kept=counts[row], compared=n,
                    exact=exact, score_exact=score_exact,
                    reference_indices=ref.view(torch.int32)[1::2][:48].tolist(),
                    packed_indices=got.view(torch.int32)[1::2][:48].tolist()))
            report["all_order_exact"] = all(x["exact"] for x in report["cases"])
            # probe 完成与候选同分合同成立分开，失败证据也正常写出。
            torch.save(output, args.output / "local_output.pt")
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("traceback", "cases")},
                     ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
