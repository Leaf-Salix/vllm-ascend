#!/usr/bin/env python3
"""只读汇总预取/量化候选的同进程三方结果，不启动设备测试。

逐作业保留五窗原始数据和实际编译策略，避免把不同设备作业的绝对差值
解释为组合收益。只有完整验证结束后才生成结果；性能分母仍为同轮原分 task。
"""

import json
from pathlib import Path


def summarize(evidence: Path) -> dict:
    """核对每个登记任务已完成，并复算中位数比值及逐窗改善数量。"""
    cases = []
    for record in json.loads((evidence / "registry.json").read_text()):
        directory = Path(record["output"])
        result = json.loads((directory / "result.json").read_text())
        matrix = json.loads((directory / "source_matrix/result.json").read_text())
        assert result["stage"] == "complete" and result["passed"], directory
        assert matrix["passed"] and matrix["same_process"] and matrix["same_root_addresses"]
        assert all(item["passed"] for item in matrix["protocol"].values())
        stats = matrix["statistics_us"]
        ref = stats["spmd_reference"]
        candidate = stats["spmd_candidate"]
        dynamic = stats["dynamic_pto"]
        assert len(ref["samples_us"]) == len(candidate["samples_us"]) == 5
        reduced = 100 * (1 - candidate["median_us"] / ref["median_us"])
        assert abs(reduced - matrix["candidate_vs_reference"]["latency_reduction_pct"]) < 1e-9
        cases.append({
            "history": record["history"], "batch": record["batch"],
            "task": record["stdout"].strip(), "output": str(directory),
            "full_validation_passed": True,
            "statistics_us": stats,
            "policies": matrix["policies"],
            "candidate_vs_reference_reduction_pct": reduced,
            "candidate_faster_windows": sum(
                candidate_us < reference_us for reference_us, candidate_us
                in zip(ref["samples_us"], candidate["samples_us"])
            ),
            "candidate_vs_dynamic_reduction_pct": 100 * (
                1 - candidate["median_us"] / dynamic["median_us"]
            ),
            "target_met": candidate["median_us"] <= 0.70 * dynamic["median_us"],
        })
    return {
        "reference_commit": "d253d627",
        "measurement": "完整L1 incore首尾；三方同进程同输入；五窗中位数；SPMD双方100 tick失败轮询退避",
        "precision_scope": "原PTO精确对照、图重放、状态/保护区及72核代际；不代表Native/token或全部动态场景验收",
        "cases": cases,
    }


def main():
    root = Path(__file__).resolve().parent
    for name in ("short_g2_prefetch_evidence", "oa_quant_reuse_evidence"):
        directory = root / name
        report = summarize(directory)
        (directory / "DEVICE_RESULTS.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        lines = [
            "# 同进程完整 CSA 五窗结果", "",
            report["measurement"], "", report["precision_scope"], "",
            "| 历史 / B | 原分 task μs | d253 SPMD μs | 候选 μs | 相对 d253 降低 | 更快窗口 | 达到30%目标 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for case in report["cases"]:
            stats = case["statistics_us"]
            lines.append(
                f"| {case['history']} / {case['batch']} | "
                f"{stats['dynamic_pto']['median_us']:.2f} | "
                f"{stats['spmd_reference']['median_us']:.2f} | "
                f"{stats['spmd_candidate']['median_us']:.2f} | "
                f"{case['candidate_vs_reference_reduction_pct']:.3f}% | "
                f"{case['candidate_faster_windows']}/5 | "
                f"{'是' if case['target_met'] else '否'} |"
            )
        lines.extend(["", "原始五窗、实际策略、任务及完整结果路径见同目录 DEVICE_RESULTS.json。", ""])
        (directory / "DEVICE_RESULTS.md").write_text("\n".join(lines))
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
