# SPDX-License-Identifier: Apache-2.0
"""从已完成的配对诊断导出阶段完成端点；不启动设备，不推算纯计算耗时。

只选择静态 layout 中每核末成员都确实发布的阶段，避免把部分发布者当作
全阶段完成。每轮先取全部发布者的最晚端点，再报告三轮中位数和原始样本。
不同阶段完成点之差仍含等待、搬运和插桩，不能作为某个 helper 的净耗时。
"""

import argparse
import json
import statistics
from pathlib import Path


ENDPOINTS = (
    ("HC 线性投影", "PARTIAL", "hc_pre_linear"),
    ("HC 归一化输出", "NORM", "mix_x_rms_norm"),
    ("QA 投影", "QA", "qr_proj_matmul"),
    ("Indexer query RoPE", "IDX_QUERY_ROPE", "idx_qr_dequant_rope"),
    ("Indexer query 量化", "IDX_QUERY_QUANT", "qr_hadamard_quant"),
    ("主 QB 投影", "QB", "qproj_matmul"),
    ("Indexer Score AIC", "SCORE_CUBE", "indexer_score_topk_native_pair_aic"),
    ("Indexer Score AIV", "SCORE_VECTOR", "indexer_score_topk_native_pair_aiv"),
    ("Top-K 合并", "TOPK", "indexer_topk_query_merge__2"),
    ("Attention AIV", "ATTN_VECTOR", "qk_pv_aiv"),
    ("输出 OB 投影", "OB", "_proj_b_mm_nz_kernel__2"),
)


def analyze(directory):
    """读取冻结的原始时钟；保留所有轮次，不挑选较快样本。"""
    result = json.loads((directory / "result.json").read_text())
    assert result["passed"] and result["full_csa_validated"]
    assert result["graph_replay_validated"] and result["dynamic_plus_static"]
    assert not result["eligible_for_performance"]
    coverage_path = Path(__file__).parent / "STATIC_ENDPOINT_COVERAGE.json"
    coverage = {
        row["kernel"]: row
        for row in json.loads(coverage_path.read_text())["kernels"]
    }
    rows_by_side = {
        side: [
            json.loads((directory / f"clock_round_{r}_{side}/phase_clock.json").read_text())
            for r in range(result["rounds"])
        ]
        for side in ("spmd_reference", "static_reference")
    }
    endpoints = []
    for label, dynamic_name, static_name in ENDPOINTS:
        assert coverage[static_name]["complete_endpoint"], static_name
        samples = {}
        for side, name in (("spmd_reference", dynamic_name), ("static_reference", static_name)):
            values = []
            for rows in rows_by_side[side]:
                selected = [row["notify_start_us"] for row in rows if row["name"] == name]
                assert selected, (side, name)
                values.append(max(selected))
            samples[side] = values
        dynamic_us = statistics.median(samples["spmd_reference"])
        static_us = statistics.median(samples["static_reference"])
        endpoints.append({
            "label": label,
            "dynamic_phase": dynamic_name,
            "static_kernel": static_name,
            "dynamic_samples_us": samples["spmd_reference"],
            "static_samples_us": samples["static_reference"],
            "dynamic_median_us": dynamic_us,
            "static_median_us": static_us,
            "difference_of_medians_us": dynamic_us - static_us,
            "static_endpoint_complete": True,
        })
    report = {
        "result": str(directory / "result.json"),
        "task_id": "task_20261018_152604_395917310445",
        "eligible_for_performance": False,
        "input_sequence": result["input_sequence"],
        "clock_records_per_round": {
            side: [len(rows) for rows in rounds] for side, rounds in rows_by_side.items()
        },
        "limitations": [
            "只诊断128K/B16/S6，不外推其它档位；基于正式权重与合成历史。",
            "时刻从各侧最早BEGIN发布点起算，包含核启动偏斜，未测完整L1。",
            "两侧每次发布都加相同四字时钟，但发布次数不同，扰动不等量。",
            "端点差含计算、等待、搬运、缓存维护和插桩，不能精确拆分未插桩的96.62微秒。",
            "静态Attention AIC、最终HC_post没有完整发布端点，未填造其耗时。",
            "本次只有完成端点；不能区分Attention的入场等待与核内流水各占多少。",
        ],
        "endpoints": endpoints,
    }
    (directory / "ENDPOINT_COMPARISON.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 静态/动态 CSA 阶段完成端点对齐",
        "",
        "任务 `task_20261018_152604_395917310445`，128K/B16/S6，A→B→A 三轮。",
        "完整数值、Top-K、score、cache/state、图重放和原有保护检查通过。",
        "以下为插桩完成时刻，不是阶段净耗时，也不是完整 L1 性能。",
        "",
        "| 完成端点 | 静态三轮中位 / μs | 动态三轮中位 / μs | 动态减静态 / μs |",
        "|---|---:|---:|---:|",
    ]
    for row in endpoints:
        lines.append(
            f"| {row['label']} | {row['static_median_us']:.2f} | "
            f"{row['dynamic_median_us']:.2f} | {row['difference_of_medians_us']:+.2f} |"
        )
    lines.extend(["", "## 解读边界", ""])
    lines.extend(f"- {text}" for text in report["limitations"])
    lines.extend([
        "",
        "动态每轮1619个时钟记录，静态794个，不能假定探针对两侧开销相等。",
        "静态完整覆盖依据见同级上层 `STATIC_ENDPOINT_COVERAGE.json`。",
        "仅有部分发布者的阶段没有进入此表；原始三轮数据完整保留在 JSON。",
    ])
    (directory / "ENDPOINT_COMPARISON.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    summary = analyze(arguments.directory.resolve())
    print(json.dumps({"endpoints": len(summary["endpoints"]), "eligible_for_performance": False}))
