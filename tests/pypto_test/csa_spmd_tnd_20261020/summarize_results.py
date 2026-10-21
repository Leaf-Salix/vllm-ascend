# SPDX-License-Identifier: Apache-2.0
"""Summarize saved evidence without importing Torch or allocating an NPU."""

import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    jobs = json.loads((here / "jobs.json").read_text())
    cases, performance = [], []
    for job in jobs:
        directory = Path(job["output"])
        if not directory.is_absolute():
            directory = here / "runs" / directory.name
        path = directory / "result.json"
        if not path.exists():
            cases.append(dict(task_id=job["task_id"], output=directory.name, missing_result=True))
            continue
        result = json.loads(path.read_text())
        snapshot = directory.with_name(directory.name + "_source")
        baseline_manifest = snapshot / "bsh_baseline_manifest.json"
        baseline_commit = None
        baseline_local_fix = None
        if baseline_manifest.exists():
            baseline_info = json.loads(baseline_manifest.read_text())["bsh"]
            baseline_commit = baseline_info["commit"]
            baseline_local_fix = baseline_info.get("local_compile_fix")
        elif result.get("bsh_wait_policy_audit", {}).get("passed"):
            # Older launchers froze source hashes without copying the manifest.
            expected = json.loads((here / "bsh_baseline_manifest.json").read_text())
            sources = json.loads((snapshot / "source_manifest.json").read_text())
            frozen = {
                "original_spmd/" + key.removeprefix("deepseek_v4_flash_csa_spmd/"): value
                for key, value in expected["bsh"]["files"].items()
                if key.startswith("deepseek_v4_flash_csa_spmd/")
            }
            if all(sources.get(key) == value for key, value in frozen.items()):
                baseline_commit = expected["bsh"]["commit"]
        row = dict(
            task_id=job["task_id"],
            output=directory.name,
            queue_status=job.get("last_observed_status"),
            stage=result.get("stage"),
            reported_passed=result.get("passed"),
            completed_pass=(
                job.get("last_observed_status") == "completed (exit=0)"
                and result.get("stage") == "complete"
                and result.get("passed") is True
            ),
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_snapshot=str(directory) + "_source",
            error=result.get("error"),
            raw_wait=result.get("wait_backoff") is False,
            native_frozen_output=result.get("native_frozen_output"),
            original_spmd_attribution=result.get("original_spmd_attribution", {}).get("candidate_exactly_matches"),
            baseline_kind=(
                "BSH150@" + (baseline_commit[:8] if baseline_commit else "未识别版本")
                + ("+静态形状修复" if baseline_local_fix else "")
                if result.get("bsh_wait_policy_audit", {}).get("passed")
                else "raw TWAIT" if result.get("wait_backoff") is False else "legacy probe"
            ),
            baseline_commit=baseline_commit,
            baseline_local_compile_fix=baseline_local_fix,
            wait_policy_experiment=result.get("wait_policy_experiment"),
            constant_indexer_history=result.get("constant_indexer_history", False),
        )
        for key in ("batch", "history", "query_lengths", "history_lengths", "token_capacity", "actual_tokens"):
            row[key] = result.get(key)
        if result.get("service_route"):
            service = result["service_route"]
            row["service"] = {
                key: service.get(key)
                for key in (
                    "passed",
                    "framework_acl_graph_wrapper",
                    "workspace_reused_across_capture",
                    "submissions",
                )
            }
            row["service"]["scope"] = service.get("scope")
            row["service"]["changed_metadata_graph"] = (
                service.get("changed_metadata_graph", {}).get("passed")
                if service.get("changed_metadata_graph")
                else None
            )
            if service.get("service_cost"):
                row["service_cost"] = {
                    key: service["service_cost"][key]
                    for key in ("passed", "scope", "median_us", "service_minus_direct_us")
                }
        if result.get("metadata_matrix"):
            row["metadata_matrix"] = {
                "passed": result["metadata_matrix"].get("passed"),
                "independent_cases": [
                    {"name": item.get("name"), "passed": item["passed"]}
                    for item in result["metadata_matrix"].get("metadata_cases", [])
                ],
                "continuous_steps": [
                    {"passed": item["passed"]} for item in result["metadata_matrix"].get("continuous_steps", [])
                ],
            }
        cases.append(row)
        if result.get("source_matrix", {}).get("statistics_us"):
            stats = result["source_matrix"]["statistics_us"]
            old, new = stats["spmd_reference"]["median_us"], stats["spmd_candidate"]["median_us"]
            performance.append(
                dict(
                    case=f"{result['history'] // 1024}K/B{result['batch']}/S6"
                    + (" G6试验（未采用）" if "g6_candidate" in directory.name else ""),
                    task_id=job["task_id"],
                    result=f"runs/{directory.name}/result.json",
                    completed_pass=row["completed_pass"],
                    statistics_us=stats,
                    original_us=old,
                    tnd_us=new,
                    change_pct=(new / old - 1) * 100,
                    multitask_us=stats["dynamic_pto"]["median_us"],
                    baseline_kind=row["baseline_kind"],
                )
            )
        elif result.get("l1", {}).get("statistics_us"):
            stats = result["l1"]["statistics_us"]
            old, new = stats["spmd_original"]["median_us"], stats["spmd_tnd"]["median_us"]
            performance.append(
                dict(
                    case=(
                        f"{result['history'] // 1024}K/B{result['batch']}/S6"
                        if result["query_lengths"] == [6] * result["batch"]
                        else f"mixed {result['query_lengths']} / {result['history'] // 1024}K"
                    )
                    + (" 共享工作区归因" if result["l1"].get("shared_workspace") else " 独立工作区"),
                    task_id=job["task_id"],
                    result=f"runs/{directory.name}/result.json",
                    completed_pass=row["completed_pass"],
                    statistics_us=stats,
                    original_us=old,
                    tnd_us=new,
                    change_pct=(new / old - 1) * 100,
                    baseline_kind=row["baseline_kind"],
                    candidate=directory.name,
                )
            )
    summary = {
        "acceptance": (
            "TND语义及无新增回归；既有Native数值差异原样单列；"
            "另需同一冻结候选的八档S6相对BSH150的等权几何平均耗时比<=0.90"
        ),
        "metric": "全72核最早incore开始至最晚结束，包含等待及阶段空隙，单位us",
        "cases": cases,
        "performance": performance,
    }
    selection_path = here / "latest_complete_cohort.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    cohort_path = here / (selection["report"] if selection else "cohort_prefix_typefix_v47_result.json")
    cohort = json.loads(cohort_path.read_text()) if cohort_path.exists() else None
    if cohort is not None:
        summary["latest_complete_s6_cohort"] = {
            "report": cohort_path.name,
            "geometric_mean_ratio": cohort["geometric_mean_ratio"],
            "threshold": cohort["threshold"],
            "passed": cohort["passed"],
            "cases": [
                {key: row[key] for key in ("history", "batch", "task_id", "bsh_us", "tnd_us", "latency_change_pct")}
                for row in cohort["cases"]
            ],
        }
    (here / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 本轮设备证据汇总",
        "",
        "由 `summarize_results.py` 从原始结果生成。只有queue exit 0、stage complete、",
        "passed true同时满足才记整项通过；原始FAIL、异常及局部通过证据保留。",
        "基线列区分旧raw TWAIT回归与另一会话已采用的BSH150。候选试验按原始目录",
        "分别保留；不能把不同源码或轮次的最快值拼成八档。只有同一冻结候选完整",
        "八档经 `evaluate_s6_cohort.py` 核验，才能判定是否满足几何平均<=0.90。",
        "当前正式参考更新为a241fa61；旧基线数据保留版本标记。",
        "8K未启用B1分支的静态形状修复单独标记；原始源码和编译FAIL保留。",
        "B1 G6/24叶方案的v31随机输入收益已被v35同分TopK/输出回归否决，不能采用。",
        "等待前四次快轮询属于测试源码改写诊断，不能用于最终八档性能验收。",
        "",
    ]
    if cohort is not None:
        lines += [
            "## 最新完整八档",
            "",
            selection["description"] if selection else "历史候选的源码与复用范围见报告。",
            f"完整证据：[{cohort_path.name}]({cohort_path.name})。",
            f"等权GM(TND/BSH)={cohort['geometric_mean_ratio']:.9f}，门槛{cohort['threshold']:.2f}，",
            f"性能验收为{'PASS' if cohort['passed'] else 'FAIL（未达标）'}。后续单项试验不替换本轮任一档。",
            "",
            "| 场景 | BSH µs | TND µs | 耗时变化 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in cohort["cases"]:
            lines.append(
                f"| {row['history'] // 1024}K/B{row['batch']}/S6 | {row['bsh_us']:.2f} | "
                f"{row['tnd_us']:.2f} | {row['latency_change_pct']:+.2f}% |"
            )
        lines.append("")
    lines += [
        "## 完整L1性能",
        "",
        "五个轮换窗口的中位数，单位µs。百分比为TND相对原单SPMD的耗时变化；",
        "正值更慢。计全72核首尾，包含等待和阶段间隙。多task只参与其支持的S6。",
        "",
        "| 场景 | 基线 | 原SPMD | TND | 变化 | 多task | 本任务功能/采样 | 原始结果 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in performance:
        multitask = f"{row['multitask_us']:.2f}" if "multitask_us" in row else "不适用"
        lines.append(
            f"| {row['case']} | {row['baseline_kind']} | {row['original_us']:.2f} | {row['tnd_us']:.2f} | "
            f"{row['change_pct']:+.2f}% | {multitask} | {'PASS' if row['completed_pass'] else '未完成'} | "
            f"[结果]({row['result']}) |"
        )
    lines += [
        "",
        "## 任务与原始证据",
        "",
        "| 任务 | 用例 | 队列终态 | 结果阶段 | 整项通过 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in cases:
        lines.append(
            f"| `{row['task_id']}` | [{row['output']}](runs/{row['output']}/result.json) | "
            f"{row.get('queue_status', '未知')} | {row.get('stage', '缺少结果')} | "
            f"{'PASS' if row.get('completed_pass') else '否'} |"
        )
    lines += [
        "",
        "各任务完整命令和固定launcher SHA256见 `jobs.json`。",
        "Native超阈值元素、Top-K替换、cache/state及保护区明细仍在原始result.json；",
        "本表的PASS不表示修复了原单SPMD对Native的已有数值差异。",
        "",
    ]
    (here / "RESULTS.md").write_text("\n".join(lines))
    print(f"wrote {len(cases)} device records, {len(performance)} performance rows")


if __name__ == "__main__":
    main()
