# SPDX-License-Identifier: Apache-2.0
"""设备端完整阈值压缩/排序/归并微验证；不以此替代整层 L1 性能。

与旧语义 probe 不同，本用例真正调用新 gather_compare 和完整候选 helper，
逐 bit 比较原排序＋incoming 优先归并，覆盖保留数的排序区间边界、同分、
零候选、负阈值、旧根未满以及同地址图重放。
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import pypto.language as pl
from topk_threshold_stages import sort_top512, threshold_merge_top512

ROWS = pl.dynamic("TOPK_COMPACT_PROBE_ROWS")


@pl.jit.incore
def compact_rows(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    old: pl.Tensor[[ROWS, 1024], pl.FP32],
    valid: pl.Tensor[[ROWS], pl.INT32],
    ref: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
    candidate: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
):
    for row in pl.range(pl.tensor.dim(src, 0)):
        values = pl.load(src, [row, 0], [1, 2048])
        root = pl.load(old, [row, 0], [1, 1024])
        n = pl.read(valid, [row])
        indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), 8192)
        full_sorted = sort_top512(values, indices, n)
        tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
        full_merge = pl.tile.mrgsort(full_sorted, root, tmp=tmp)
        full_top = pl.tile.extract(full_merge, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec)
        pl.store(full_top, [row, 8], ref)
        compact_top = threshold_merge_top512(values, root, pl.cast(8192, pl.INT32), n)
        pl.store(compact_top, [row, 8], candidate)
    return ref, candidate


@pl.jit
def entry(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    old: pl.Tensor[[ROWS, 1024], pl.FP32],
    valid: pl.Tensor[[ROWS], pl.INT32],
    ref: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
    candidate: pl.InOut[pl.Tensor[[ROWS, 1040], pl.FP32]],
):
    with pl.spmd(1):
        ref, candidate = compact_rows(src, old, valid, ref, candidate)
    return ref, candidate


def fixtures(torch, seed):
    gen = torch.Generator().manual_seed(seed)
    cases, inputs, roots, valid = [], [], [], []
    for n in (513, 1024, 1025, 2048):
        for kept in sorted(set((0, 1, 31, 511, 512, min(513, n), min(1024, n), n))):
            for threshold in (-1.0, 0.5):
                values = torch.full((2048,), 12345.0)  # 无效尾 deliberately 高分，不能进入结果。
                values[:n] = threshold - 1
                positions = torch.randperm(n, generator=gen)[:kept]
                values[positions] = threshold + torch.randint(0, 3, (kept,), generator=gen).float()
                pairs = torch.empty(1024)
                pairs[0::2] = torch.linspace(threshold + 2, threshold, 512)
                pairs.view(torch.int32)[1::2] = torch.arange(512, dtype=torch.int32)
                cases.append(dict(valid=n, kept=kept, threshold=threshold))
                inputs.append(values)
                roots.append(pairs)
                valid.append(n)
    values = torch.rand(2048, generator=gen)
    pairs = torch.full((1024,), -3.4028234663852886e38)
    pairs[0::2][:511] = 1
    pairs.view(torch.int32)[1::2] = torch.arange(512, dtype=torch.int32)
    cases.append(dict(valid=2048, old_root_not_full=True))
    inputs.append(values)
    roots.append(pairs)
    valid.append(2048)
    return cases, torch.stack(inputs), torch.stack(roots), torch.tensor(valid, dtype=torch.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--device", type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto import passes, pypto_core
    from pypto.runtime import CompileOptions, ensure_pto_isa_root
    config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"),
                            dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
    report = dict(passed=False, cpu_only=args.cpu_only, cases=[], full_csa_validated=False,
                  performance_measured=False, core=pypto_core.__file__)
    try:
        report["isa"] = ensure_pto_isa_root()
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
            names, src_cpu, old_cpu, valid_cpu = fixtures(torch, 1024)
            device = f"npu:{dev}"
            src, old, valid = (x.to(device) for x in (src_cpu, old_cpu, valid_cpu))
            ref = torch.full((len(names), 1040), -7777.0, device=device)
            cand = ref.clone()
            entry(src, old, valid, ref, cand, config=config)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                entry(src, old, valid, ref, cand, config=config)
            for seed in (1024, 2048, 1024):
                names, src_cpu, old_cpu, valid_cpu = fixtures(torch, seed)
                src.copy_(src_cpu)
                old.copy_(old_cpu)
                valid.copy_(valid_cpu)
                ref.fill_(-7777)
                cand.fill_(-7777)
                graph.replay()
                torch.npu.synchronize()
                expected, actual = ref.cpu(), cand.cpu()
                failures = []
                for row, name in enumerate(names):
                    same = torch.equal(expected[row].view(torch.int32), actual[row].view(torch.int32))
                    if not same:
                        failures.append(dict(row=row, **name))
                guards = all(torch.all(x[:, :8] == -7777) and torch.all(x[:, 1032:] == -7777)
                             for x in (expected, actual))
                inputs_ok = torch.equal(src.cpu(), src_cpu) and torch.equal(old.cpu(), old_cpu)
                inputs_ok = inputs_ok and torch.equal(valid.cpu(), valid_cpu)
                report["cases"].append(dict(seed=seed, rows=len(names), failures=failures,
                                            guards=bool(guards), inputs_unchanged=inputs_ok))
                if failures or not guards or not inputs_ok:
                    torch.save(dict(input=src_cpu, old=old_cpu, valid=valid_cpu,
                                    ref=expected, candidate=actual), args.output / "local_failure.pt")
                    raise AssertionError(f"seed={seed}, failing rows={failures[:5]}")
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
