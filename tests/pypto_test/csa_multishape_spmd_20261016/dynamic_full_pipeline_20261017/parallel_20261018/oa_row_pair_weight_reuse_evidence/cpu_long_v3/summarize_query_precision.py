# SPDX-License-Identifier: Apache-2.0
"""离线汇总真实query验证，分开报告原PTO等价、Native差异和运行稳定性。

只读取已保存的CPU Tensor/JSON，不执行NPU，不修改冻结门槛。S1/S5通过
原分task真实S6的因果前缀对照，但这不能替代Native或整网输出token验收。
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # 仅离线CPU统计；调用方使用既有环境关闭设备后端自动加载。
    import torch

    rows = []
    for job in json.loads(args.jobs.read_text()):
        folder = Path(job["output"])
        result = json.loads((folder / "result.json").read_text())
        causal = result.get("causal_extension", {})
        row = dict(
            task_id=job["task_id"], history=job["history"], batch=job["batch"],
            query_tokens=job.get("query_tokens", result.get("query_tokens")),
            stage=result.get("stage"), error=result.get("error"),
            functional_passed=result.get("functional_passed", False),
            native_exact=result.get("precision_exact", False),
            original_pto_causal_exact=causal.get("passed", False),
            graph_rounds=[r["passed"] for r in result.get("rounds", [])],
            native_guards=result.get("native_guards"),
            spmd_guards=result.get("eager_guards"),
            native_output=result.get("native_comparison", {}).get("x_out"),
            native_topk_selection=result.get("native_topk_selection"),
            comparison_failures=[dict(sign=r["extra_query_sign"], key=k, result=v)
                                 for r in causal.get("rounds", [])
                                 for k, v in r["comparisons"].items() if v["status"] != "PASS"],
        )
        diagnostic = folder / "numerical_diagnostic.pt"
        if diagnostic.exists():
            tensors = torch.load(diagnostic, map_location="cpu", weights_only=True)
            actual, expected = tensors["spmd_output"].double(), tensors["native_output"].double()
            assert actual.shape == expected.shape
            finite = torch.isfinite(actual) & torch.isfinite(expected)
            difference = (actual - expected).abs()
            # 沿用既有0.01/0.01输出门槛，绝不因原PTO等价而放宽。
            mismatch = ~finite | (difference > 0.01 + 0.01 * expected.abs())
            row["native_frozen_output"] = dict(
                atol=0.01, rtol=0.01, mismatches=int(mismatch.sum()), elements=actual.numel(),
                nonfinite=int((~finite).sum()), passed=bool(not mismatch.any()),
                max_abs=float(difference.max()), rmse=float(difference.square().mean().sqrt()),
            )
        rows.append(row)
    report = dict(
        scope="真实query单层正确性；不测性能，不代表整网token通过",
        source_jobs=str(args.jobs.resolve()), cases=rows,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: row.get(k) for k in (
        "history", "batch", "query_tokens", "stage", "error", "functional_passed",
        "original_pto_causal_exact", "native_exact", "native_frozen_output",
    )} for row in rows], ensure_ascii=False))


if __name__ == "__main__":
    main()
