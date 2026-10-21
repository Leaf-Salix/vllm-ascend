# SPDX-License-Identifier: Apache-2.0
"""从现有完整结果提取CSA与task核内统计；不运行设备、不拼接跨轮收益。"""

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stats(values):
    return dict(
        samples=len(values),
        min_us=min(values),
        mean_us=statistics.mean(values),
        max_us=max(values),
        median_us=statistics.median(values),
    )


def task_name(name):
    name = name.lstrip("_")
    if name.startswith("indexer_score_topk_native_pair"):
        return "score_aic" if "_aic" in name else "score_aiv"
    if name.startswith("qr_proj_matmul"):
        return "qa_aic"
    for prefix, label in (
        ("output_cube", "output_mix_aic"),
        ("output_vector", "output_mix_aiv"),
        ("hc_row_cube", "hc_prefix_aic"),
        ("hc_row_vector", "hc_prefix_aiv"),
        ("attention_rows_cube", "attention_plan_mix_aic"),
        ("attention_rows_vector", "attention_plan_mix_aiv"),
        ("attention_plan_rows_pipeline_aic", "attention_plan_mix_aic"),
        ("attention_plan_rows_pipeline_aiv", "attention_plan_mix_aiv"),
    ):
        if name.startswith(prefix):
            return label
    if name.startswith("qk_pv_"):
        if "_aic" in name:
            return "qk_pv_aic"
        if "_aiv" in name:
            return "qk_pv_aiv"
    for prefix in ("qk_pv_aic", "qk_pv_aiv", "proj_b_mm", "qproj_matmul", "qproj_dequant_rms_nope_rope", "proj_a_mm"):
        if name.startswith(prefix):
            return prefix
    return None


def summarize(path):
    report = json.loads(path.read_text())
    summary = {
        "result": str(path.relative_to(HERE)),
        "stage": report["stage"],
        "passed": report.get("passed", False),
        "candidate": report["candidate"],
        "candidate_wait_backoff": report.get("candidate_wait_backoff", "off"),
        "candidate_wait_units": report.get("candidate_wait_units", "score"),
        "current": report.get("current"),
        "indexer_page_pattern": report.get("indexer_page_pattern", "original"),
        "batch": report["batch"],
        "history": report["history"],
        "profiling": report.get("profiling", {"chip_swimlane_level": 1, "dep_gen": True}),
    }
    if not summary["passed"]:
        summary["error"] = report.get("error", "").splitlines()[-1:]
        return summary
    if "l1" not in report:
        summary.update(
            performance_measured=False,
            functional_graph_checks=len(report.get("functional_graph", {})),
            boundary_replay=report.get("boundary_replay", {}).get("passed"),
        )
        return summary
    l1 = report["l1"]
    summary["full_csa_l1_us"] = l1["statistics_us"]
    if "outer" in l1:
        summary["outer_graph"] = l1["outer"]
    if "boundary_replay" in report:
        summary["boundary_replay"] = {
            "passed": report["boundary_replay"]["passed"],
            "same_graph_replays": report["boundary_replay"].get("same_graph_replays"),
            "result": str(path.parent.relative_to(HERE) / "boundary_replay.json"),
        }
    summary["candidate_reduction_vs_original_pct"] = l1["candidate_reduction_vs_original_pct"]
    summary["candidate_reduction_vs_spmd_pct"] = l1["candidate_reduction_vs_spmd_pct"]
    if "candidate_reduction_vs_current_pct" in l1:
        summary["candidate_reduction_vs_current_pct"] = l1["candidate_reduction_vs_current_pct"]
    summary["incore_scope"] = (
        "每物理核task执行时长，含核内等待/握手；不包含receive_to_start，"
        "不等于纯Cube/Vector运算，也不能把各核时间相加当作关键路径收益"
    )
    if report["candidate"].startswith("fused_topk") and report["history"] > 32768 and report["batch"] >= 4:
        summary["candidate_scope_change"] = (
            "候选Score AIV还包含根发布/等待和最终Top-K；不能与原分离版Score AIV直接作等工作量核内比较"
        )
    if report["candidate"].startswith("fused_wait"):
        summary["candidate_scope_change"] = (
            "候选Score AIV包含Top-K及信号等待；等待源码策略见candidate_wait_backoff和candidate_wait_codegen/audit.json"
        )
    if report["candidate"].startswith("query_score"):
        summary["candidate_scope_change"] = (
            "长G6拆M的Score AIV包含Q反量化/RMS/RoPE；与原Score不能按等工作量相减，"
            "完整CSA及QB结束到Attention首启另列；短档或小批仍是独立Q任务"
        )
        if (report.get("current") or "").startswith("query_score"):
            summary["current_scope_change"] = (
                "当前对照同样是未采用的Q融合候选，两者Score工作范围相同；原始多task的Score仍不含Q，不能混作同工作量对照"
            )
    if report["candidate"].startswith(("output_groups", "hc_rows", "attention_rows")):
        summary["candidate_scope_change"] = (
            "局部MIX核内时间合并多个数学阶段与软等待；与原分离task不能按同名数学直接相减"
        )
    timings, per_window = {}, []
    for window in l1["windows"]:
        if window["side"] == "single_spmd":
            continue
        trace = json.loads(Path(window["trace"]).read_text())
        groups = {}
        for event in trace["traceEvents"]:
            name = task_name(event.get("name", ""))
            duration = event.get("args", {}).get("kernel-duration-us")
            if event.get("ph") != "X" or name is None or duration is None:
                continue
            groups.setdefault(name, []).append(duration)
            timings.setdefault(window["side"], {}).setdefault(name, []).append(duration)
        per_window.append(
            dict(side=window["side"], trace=window["trace"], tasks={name: stats(v) for name, v in groups.items()})
        )
    summary["task_incore_us"] = {
        side: {name: stats(v) for name, v in groups.items()} for side, groups in timings.items()
    }
    summary["task_windows"] = per_window
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.results or sorted((HERE / "runs").glob("*/h*/result.json"))
    results = [summarize(path.resolve()) for path in paths]
    # 每窗的原始trace及统计仍保留在runs；提交汇总只留完整CSA五窗和task聚合，
    # 避免为每个历史case重复展开相同的几十组task字段。
    for result in results:
        result.pop("task_windows", None)
    destination = HERE / "RESULTS.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("task_windows", "task_incore_us", "incore_scope")},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
