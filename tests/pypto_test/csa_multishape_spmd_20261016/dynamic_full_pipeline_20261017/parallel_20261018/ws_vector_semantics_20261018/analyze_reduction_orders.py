#!/usr/bin/env python3
"""对已捕获单K384 Cube结果做CPU浮点归约顺序核对；不编译或再占NPU。

每个FP16乘积在FP32中可精确表示。这里只比较常见分组/加法顺序，不声称
穷举了硬件实现；即使有限样本一致，也不能直接替代设备Vector实现验证。
"""

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def reduce_values(values, mode):
    if mode == "sequential":
        total = np.zeros(values.shape[:-1], dtype=np.float32)
        for index in range(values.shape[-1]):
            total = np.add(total, values[..., index], dtype=np.float32)
        return total
    while values.shape[-1] > 1:
        if mode == "adjacent":
            values = np.add(values[..., 0::2], values[..., 1::2], dtype=np.float32)
        else:
            half = values.shape[-1] // 2
            values = np.add(values[..., :half], values[..., half:], dtype=np.float32)
    return values[..., 0]


def main():
    results = []
    for file in sorted((HERE / "device_single_k384_v1").glob("*.pt")):
        data = torch.load(file, weights_only=True, map_location="cpu")
        coeff = data["coeff"].float().numpy()
        qk = data["qk"].float().numpy()
        weights = np.stack([coeff[q, q * 64:(q + 1) * 64] for q in range(6)])
        panels = qk.reshape(6, 64, 64).transpose(0, 2, 1)
        products = np.multiply(panels, weights[:, None, :], dtype=np.float32)
        cube = data["cube"].numpy()
        row = dict(case=file.stem, variants=[])
        for size in (2, 4, 8, 16, 32, 64):
            chunks = products.reshape(6, 64, 64 // size, size)
            for inner in ("sequential", "adjacent", "strided"):
                partial = reduce_values(chunks, inner)
                for outer in ("sequential", "adjacent", "strided"):
                    actual = reduce_values(partial, outer)
                    difference = int(np.count_nonzero(actual.view(np.uint32) != cube.view(np.uint32)))
                    row["variants"].append(dict(chunk=size, inner=inner, outer=outer,
                                                bit_differences=difference,
                                                max_abs=float(np.max(np.abs(actual - cube)))))
        row["exact_variants"] = [v for v in row["variants"] if v["bit_differences"] == 0]
        results.append(row)
        print(file.stem, "exact", len(row["exact_variants"]), "best differences",
              min(v["bit_differences"] for v in row["variants"]))
    nontrivial = [r for r in results if r["case"] != "exact_binary_control"]
    common = set.intersection(*[
        {(v["chunk"], v["inner"], v["outer"]) for v in r["exact_variants"]} for r in nontrivial
    ])
    report = dict(source="device_single_k384_v1", arithmetic="FP32逐次舍入，CPU探查有限归约顺序",
                  common_exact_variants=sorted(common), cases=results,
                  limitation="没有枚举所有硬件算法，不宣称不存在其它精确方案；未测性能")
    (HERE / "REDUCTION_ORDER_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("common exact orders:", sorted(common))


if __name__ == "__main__":
    main()
