# SPDX-License-Identifier: Apache-2.0
"""只分析已有失败快照，不重新运行设备，不据统计量自行放宽精度门槛。"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.result_directory
    saved = torch.load(directory / "failure_states.pt", map_location="cpu",
                       weights_only=True, mmap=True)
    actual = saved["candidate"]["x_out"]
    reference = saved["reference"]["x_out"]
    a, b = actual.double(), reference.double()
    difference = (a - b).abs()
    changed = actual != reference
    rows = difference.reshape(difference.shape[0], -1)
    bf16_step_distance = (actual.view(torch.int16).int() -
                          reference.view(torch.int16).int()).abs()
    # 同号 BF16 的相邻编码是相邻可表示值；跨符号不把编码差当 ULP。
    same_sign = torch.signbit(actual) == torch.signbit(reference)
    report = {
        "scope": "相对原多task PTO输出的差异统计；没有Native、FP64完整参考或整模型token结论",
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "elements": actual.numel(),
        "changed": int(changed.sum()),
        "changed_fraction": float(changed.double().mean()),
        "max_abs": float(difference.max()),
        "rmse": float(difference.square().mean().sqrt()),
        "reference_rms": float(b.square().mean().sqrt()),
        "relative_l2": float((a - b).norm() / b.norm()),
        "nonfinite": int((~torch.isfinite(a) | ~torch.isfinite(b)).sum()),
        "max_abs_by_token_row": rows.max(dim=1).values.tolist(),
        "absolute_error_quantiles": {
            str(q): float(torch.quantile(difference.flatten(), q)) for q in (0.5, 0.9, 0.99, 0.999)
        },
        "changed_same_sign_bf16_distance": {
            str(step): int((changed & same_sign & (bf16_step_distance <= step)).sum())
            for step in (1, 2, 4, 8)
        },
        "changed_across_sign": int((changed & ~same_sign).sum()),
        "frozen_bitwise_gate": "FAIL",
        "token_equality_validated": False,
        "adopted": False,
    }
    (directory / "numerical_difference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "max_abs_by_token_row"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
