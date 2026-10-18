# SPDX-License-Identifier: Apache-2.0
"""汇总已结束七档，不触发编译/设备；历史静态与本轮配对数据分开标注。"""

import json
from pathlib import Path


def summarize():
    here = Path(__file__).resolve().parent
    history_root = here.parents[2]
    registry = json.loads((history_root / "selected_shape_policies.json").read_text())
    historical = {(r["history"], r["batch"]): r for r in registry["rows"]}
    submissions = json.loads((here / "SUBMISSIONS_V1.json").read_text())
    rows = []
    for job in sorted(submissions, key=lambda j: (-j["history"], j["batch"])):
        result_path = Path(job["output"]) / "result.json"
        result = json.loads(result_path.read_text())
        assert result["passed"] and result["full_csa_validated"]
        matrix = result["source_matrix"]
        assert matrix["passed"] and matrix["same_process"] and matrix["same_root_addresses"]
        stats = matrix["statistics_us"]
        assert all(len(item["samples_us"]) == 5 for item in stats.values())
        multi = stats["dynamic_pto"]["median_us"]
        previous = stats["spmd_reference"]["median_us"]
        candidate = stats["spmd_candidate"]["median_us"]
        old = historical.get((job["history"], job["batch"]))
        static = None
        if old:
            old_result_path = history_root / old["source"]
            assert json.loads(old_result_path.read_text())["passed"]
            static = {"l1_us": old["l1_us"], "source": str(old_result_path), "same_round": False}
        rows.append({
            "history": job["history"], "batch": job["batch"], "query_tokens": 6,
            "task_id": job["task_id"], "source": str(result_path),
            "multi_task_us": multi, "previous_dynamic_us": previous, "candidate_us": candidate,
            "vs_previous_reduction_pct": (previous - candidate) / previous * 100,
            "vs_multi_reduction_pct": (multi - candidate) / multi * 100,
            "target_us": multi * 0.7, "target_met": candidate <= multi * 0.7,
            "statistics_us": stats, "historical_static": static,
        })
    assert len(rows) == 7
    report = {
        "scope": "正式第4层W8A8，S6，合成历史；每档独立同进程三方五窗完整L1中位",
        "target_met_count": sum(r["target_met"] for r in rows), "rows": rows,
        "static_note": "静态来自已验证选择登记表，跨轮仅供参考；B28历史入口导入失败，无有效性能。",
        "b28_static_failure": str(history_root / "static_reference_h131072_b28_20261018_v1/result.json"),
    }
    (here / "SEVEN_RESULTS_V1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 前段协同重排：七档完整L1",
        "",
        "所有七档的精确输出/Top-K/score、cache/state、A→B→A、图重放、保护与代际检查通过。",
        "前三个耗时列为每档同进程三方五窗中位，单位μs；没有插桩，不挑最小样本。",
        "静态列为已验证历史登记数据，**与本轮不同轮，不能用于计算本次收益**。",
        "",
        "| 历史/B | 多task | 旧动态 | 协同候选 | 相对旧动态降低 | 相对多task降低 | 历史静态 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        static = f"{row['historical_static']['l1_us']:.2f}" if row["historical_static"] else "无有效数据"
        lines.append(
            f"| {row['history'] // 1024}K/B{row['batch']} | {row['multi_task_us']:.2f} | "
            f"{row['previous_dynamic_us']:.2f} | {row['candidate_us']:.2f} | "
            f"{row['vs_previous_reduction_pct']:.2f}% | {row['vs_multi_reduction_pct']:.2f}% | {static} |"
        )
    lines.extend([
        "", "**相对原多task降低30%的目标仍0/7。** 七档均改善旧动态实现，不等于完成目标。",
        "B8仍慢于同轮多task，短B16接近持平；其它档位相对多task的收益也很有限。",
        "B16静态登记值807.16是旧轮次；另一次同进程静态/原动态对照为792.78/889.40，",
        "不能把任一旧静态值与本轮847.98直接相减报告收益。",
        "", "所有原始五窗、完整检查、任务ID及静态来源在JSON中保留。",
        "S1/S5及专门metadata/padding验证单独记录，不用本表的S6通过代替。",
    ])
    (here / "SEVEN_RESULTS_V1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": len(rows), "target_met": report["target_met_count"]}))


if __name__ == "__main__":
    summarize()
