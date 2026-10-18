#!/usr/bin/env python3
"""逐storage核对完整72核回放，区分真实结果、工作区与保护区。

直接比较设备after与模拟回放的实际字节，不做hash。别名storage只比一次；
若不一致，再按各根Tensor的真实offset/shape/stride报告差异，不能只看最后
output便认定完整回放通过。无效尾部与保护区仍属于完整storage检查范围。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_standalone import CAPTURE, HERE


def compare_bytes(expected, actual, size):
    if not actual.is_file():
        return dict(passed=False, error="missing output", expected_bytes=size)
    if actual.stat().st_size != size:
        return dict(passed=False, error="wrong size", actual_bytes=actual.stat().st_size, expected_bytes=size)
    assert expected.stat().st_size == size
    before = np.memmap(expected, dtype=np.uint8, mode="r")
    after = np.memmap(actual, dtype=np.uint8, mode="r")
    count, first = 0, None
    for start in range(0, size, 8 * 1024 * 1024):
        unequal = before[start:start + 8 * 1024 * 1024] != after[start:start + 8 * 1024 * 1024]
        n = int(np.count_nonzero(unequal))
        count += n
        if n and first is None:
            first = start + int(np.flatnonzero(unequal)[0])
    return dict(passed=count == 0, bytes=size, different_bytes=count, first_different_byte=first)


def compare_root(row, expected, actual):
    dtype = {"torch.float32": "<f4", "torch.bfloat16": "<u2", "torch.float16": "<f2",
             "torch.int8": "i1", "torch.uint8": "u1", "torch.int32": "<i4", "torch.int64": "<i8"}[row["dtype"]]
    args = dict(shape=tuple(row["shape"]), dtype=np.dtype(dtype), offset=row["offset_bytes"],
                strides=tuple(s * row["element_bytes"] for s in row["strides"]))
    left = np.ndarray(buffer=np.memmap(expected, mode="r", dtype=np.uint8), **args)
    right = np.ndarray(buffer=np.memmap(actual, mode="r", dtype=np.uint8), **args)
    # 用相同宽度整数比bit，避免NaN自不等和+0/-0掩盖位差。
    bits = np.dtype(f"<u{row['element_bytes']}")
    count = int(np.count_nonzero(left.view(bits) != right.view(bits)))
    report = dict(bitwise_equal=count == 0, different_elements=count, shape=row["shape"], dtype=row["dtype"])
    if count:
        if row["dtype"] == "torch.bfloat16":
            left = (left.astype(np.uint32) << 16).view(np.float32)
            right = (right.astype(np.uint32) << 16).view(np.float32)
        left, right = left.astype(np.float64), right.astype(np.float64)
        finite = np.isfinite(left) & np.isfinite(right)
        report.update(finite_max_abs=float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else None,
                      finite_mismatch=int(np.count_nonzero(np.isfinite(left) != np.isfinite(right))))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "REPLAY_VALIDATION.json")
    args = parser.parse_args()
    case = args.case or Path((HERE / "ACTIVE_CASE_PATH.txt").read_text().strip())
    capture = json.loads((CAPTURE / "replay/capture.json").read_text())
    results, roots = {}, {}
    for key, row in capture["storages"].items():
        if not row["written"]:
            continue
        expected = CAPTURE / "replay/after" / (key + ".bin")
        actual = case / "actual" / (key + ".bin")
        results[key] = compare_bytes(expected, actual, row["bytes"])
        results[key]["roots"] = row["names"]
        if not results[key]["passed"] and "error" not in results[key]:
            for name in row["names"]:
                roots[name] = compare_root(capture["roots"][name], expected, actual)
    report = dict(passed=all(r["passed"] for r in results.values()), case=str(case),
                  compared_storages=len(results), compared_bytes=sum(r["bytes"] for r in capture["storages"].values() if r["written"]),
                  storage_results=results, differing_storage_roots=roots,
                  method="实际物理storage逐字节比较；无hash；包含共享别名、尾部和保护区")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "compared_storages", "compared_bytes")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
