# SPDX-License-Identifier: Apache-2.0
"""独立核对压缩计数与有序索引，防止回退完整排序掩盖未生效的筛选。

只验证本轮实际使用的 FP32/GE 路径：真实 Scalar 阈值、动态 valid_cols、
保序索引、非零 offset、GM 保护区。此结果不代替完整 CSA 的精度或性能。
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import pypto.language as pl
from topk_compact_probe import fixtures

ROWS = pl.dynamic("GATHER_DIRECT_PROBE_ROWS")


@pl.jit.incore
def gather_rows(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    valid: pl.Tensor[[ROWS], pl.INT32],
    thresholds: pl.Tensor[[ROWS], pl.FP32],
    dst: pl.InOut[pl.Tensor[[ROWS, 2064], pl.INT32]],
    counts: pl.InOut[pl.Tensor[[ROWS, 16], pl.INT32]],
):
    for row in pl.range(pl.tensor.dim(src, 0)):
        values = pl.load(src, [row, 0], [1, 2048])
        n = pl.cast(pl.read(valid, [row]), pl.INDEX)
        values = pl.set_validshape(values, 1, n)
        threshold = pl.read(thresholds, [row])
        tmp = pl.tile.create([1, 18432], dtype=pl.UINT8)
        selected, count = pl.tile.gather_compare(
            values, threshold, tmp, cmp_mode="ge", out_cols=2048, offset=17)
        pl.store(selected, [row, 8], dst)
        # 每行独占64B，只有当前一个worker写入；不与其它Scalar写者共享缓存行。
        counts = pl.write(counts, [row, 8], pl.tile.read(count, [0, 0]))
    return dst, counts


@pl.jit
def entry(
    src: pl.Tensor[[ROWS, 2048], pl.FP32],
    valid: pl.Tensor[[ROWS], pl.INT32],
    thresholds: pl.Tensor[[ROWS], pl.FP32],
    dst: pl.InOut[pl.Tensor[[ROWS, 2064], pl.INT32]],
    counts: pl.InOut[pl.Tensor[[ROWS, 16], pl.INT32]],
):
    with pl.spmd(1):
        dst, counts = gather_rows(src, valid, thresholds, dst, counts)
    return dst, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "rows": [], "full_csa_proven": False}
    try:
        import torch
        import torch_npu  # noqa: F401
        import pypto.torch
        from pypto.runtime import CompileOptions

        dev = int(os.environ["TASK_DEVICE"])
        assert args.device is None or args.device == dev
        torch.npu.set_device(dev)
        pypto.torch.init(device=dev, platform="a2a3", runtime="tensormap_and_ringbuffer")
        names, src_cpu, roots, valid_cpu = fixtures(torch, 1024)
        thresholds_cpu = roots[:, 1022].contiguous()
        src, valid, thresholds = [t.to(f"npu:{dev}") for t in
                                  (src_cpu, valid_cpu, thresholds_cpu)]
        dst = torch.full((len(names), 2064), -777, dtype=torch.int32, device=src.device)
        counts = torch.full((len(names), 16), -777, dtype=torch.int32, device=src.device)
        config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"), dump_passes=True)
        entry(src, valid, thresholds, dst, counts, config=config)
        torch.npu.synchronize()
        observed, totals = dst.cpu(), counts.cpu()
        for row, name in enumerate(names):
            n = int(valid_cpu[row])
            expected = (src_cpu[row, :n] >= thresholds_cpu[row]).nonzero().flatten().int() + 17
            count = int(totals[row, 8])
            good = count == len(expected) and torch.equal(observed[row, 8:8 + len(expected)], expected)
            report["rows"].append({**name, "observed_count": count, "expected_count": len(expected),
                                   "ordered_indices_exact": good})
        report["guards"] = bool(
            (observed[:, :8] == -777).all() and (observed[:, -8:] == -777).all()
            and (totals[:, :8] == -777).all() and (totals[:, 9:] == -777).all())
        report["inputs_unchanged"] = (torch.equal(src.cpu(), src_cpu)
                                      and torch.equal(valid.cpu(), valid_cpu)
                                      and torch.equal(thresholds.cpu(), thresholds_cpu))
        report["passed"] = (all(r["ordered_indices_exact"] for r in report["rows"])
                            and report["guards"] and report["inputs_unchanged"])
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "traceback")},
                     ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
