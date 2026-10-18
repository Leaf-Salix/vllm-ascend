#!/usr/bin/env python3
"""只汇总已有七档结果；核对原始样本中位数，不执行新的NPU测试。

正式动态SPMD使用最近测量；明确单列未采用的B16候选。每行原分task与
动态SPMD取同一结果文件。Native与静态列允许历史独立轮次，保留各自路径、
样本和版本，不能据这两列与其他列直接计算同轮加速比。
"""

import json
import statistics
from copy import deepcopy
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARALLEL = HERE / "parallel_20261018"


def read(path):
    return json.loads(path.read_text())


def checked_statistics(value):
    samples = value["samples_us"]
    median = statistics.median(samples)
    if abs(median - value["median_us"]) > 1e-8:
        raise ValueError("报告中位数与保存样本不符")
    return {"median_us": median, "samples_us": samples, "count": len(samples)}


def main():
    historical = read(HERE / "FOUR_WAY_PERFORMANCE_REFERENCE_20261018.json")
    rows = deepcopy(historical["rows"])
    for row in rows:
        if (row["history"], row["batch"]) == (131072, 16):
            paired_path = PARALLEL / "score_attention_interleave_candidate/static_same_process_h131072_b16_v1/result.json"
            row["pto_pair_task"] = "task_20261018_092130_124935114251"
        else:
            paired_path = Path(row["pto_pair_output"]) / "result.json"
        result = read(paired_path)
        if not result["passed"] or not result["performance_measured"]:
            raise ValueError(f"未完成有效性能测量：{paired_path}")
        axis = "source_matrix" if "source_matrix" in result else "l1"
        statistics_us = result[axis]["statistics_us"]
        dynamic_role = "spmd_reference" if axis == "source_matrix" else "single_spmd"
        baseline = checked_statistics(statistics_us["dynamic_pto"])
        spmd = checked_statistics(statistics_us[dynamic_role])
        row.update(
            dynamic_pto_full_l1_us=baseline["median_us"],
            single_spmd_full_l1_us=spmd["median_us"],
            dynamic_pto_samples_us=baseline["samples_us"],
            single_spmd_samples_us=spmd["samples_us"],
            pto_pair_samples=baseline["count"],
            pto_pair_output=str(paired_path.parent),
            pto_pair_source=str(paired_path),
            pto_pair_json=str(paired_path),
            pto_pair_result_axis=axis,
            pto_pair_spmd_role=dynamic_role,
            same_process_and_input_within_pto_pair=True,
            medians_recomputed_from_existing_samples=True,
            dynamic_source=result[axis].get("reference_source", result.get("source_package")),
            environment=result["environment"],
            spmd_time_change_percent=(spmd["median_us"] / baseline["median_us"] - 1) * 100,
        )
        static = row["static_spmd_historical_reference"]
        if static["median_us"] is not None:
            if (row["history"], row["batch"]) == (131072, 16):
                static_path = PARALLEL / "projection_branch_evidence/h131072_b16_branch4_v1/result.json"
                static_result = read(static_path)
                static_stats = checked_statistics(static_result["l1"]["statistics_us"]["static_spmd"])
                static.update(
                    median_us=static_stats["median_us"],
                    same_run_dynamic_pto_us=static_result["l1"]["statistics_us"]["dynamic_pto"]["median_us"],
                    source=str(static_path),
                    task="task_20261018_114949_268532326529",
                    samples_us=static_stats["samples_us"],
                )
            else:
                static_result = read(Path(static["source"]))
                static_stats = checked_statistics(static_result["l1_incore"]["single_spmd"])
                static["samples_us"] = static_stats["samples_us"]
                if static_stats["median_us"] != static["median_us"]:
                    raise ValueError("静态参考值与原结果不符")
        row["target_30_percent_met"] = spmd["median_us"] <= baseline["median_us"] * 0.7
        row["version_note"] = (
            "B28最近数据来自9d5327e5；之后66882634仅新增G6且16<T<32的QA路径，T168不选用；最新依赖环境未重测"
            if row["batch"] == 28 else "正式66882634算法的最近可用测量；后续失败候选未采用"
        )
    native_b28 = read(HERE.parents[1] / "native_h131072_b28_optimized_20261018/device_v1/summary.json")
    b28 = next(row for row in rows if row["batch"] == 28)
    assert statistics.median(native_b28["profiled_all_stream_layer_span_us"]["samples"]) == b28["native_historical_external_profiler"]["median_us"]
    failed_path = PARALLEL / "projection_branch_evidence/h131072_b16_branch4_v1/result.json"
    failed = read(failed_path)
    candidates = [{
        "case": "128K/B16/S6", "candidate": "4核辅助投影分支", "adopted": False,
        "result": str(failed_path), "task": "task_20261018_114949_268532326529",
        "statistics_us": {k: checked_statistics(v) for k, v in failed["l1"]["statistics_us"].items()},
        "reason": "完整L1比同轮多task慢7.62%，正式目录已恢复原版",
    }]
    report = {
        "date": "2026-10-18", "scope": "完整CSA含HC_pre/HC_post；正式第4层W8A8；合成历史；S6",
        "new_measurements": False,
        "metric_native": historical["native_metric"], "metric_pto": historical["pto_metric"],
        "comparability": "每行PTO多task和动态SPMD同轮；Native与静态是独立历史轮次；不同计时边界不计算相互加速比",
        "selection": "保留正式采用策略的最近结果，不取所有试验中的最小值；最新失败候选另列",
        "rows": rows, "latest_unadopted_candidates": candidates,
        "b28_data_complete_except_static": True,
        "b28_static_not_retested": "用户要求只用已有数据；既有尝试导入失败，无有效静态性能",
        "target_met_count": sum(row["target_30_percent_met"] for row in rows),
    }
    stem = HERE / "LATEST_SEVEN_CASE_PERFORMANCE_20261018"
    stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# CSA七档最新已有性能对照（2026-10-18）", "",
        "正式第4层W8A8、S=6、完整CSA含HC_pre/HC_post，单位μs，均为中位数。本次未启动新测试。", "",
        "动态列选当前正式采用策略的最近实测，不从实验候选中挑最小值。每行多task与动态SPMD同轮；静态列取已有独立轮次。", "",
        "Native是外部profiler完整设备kernel跨度，包含调度；PTO是完整L1 incore首尾，排除外围首尾。两种口径不能直接算加速比。", "",
        "| 历史 / Batch | Native外部profiling | PTO多task L1 | 静态SPMD L1（历史） | 动态单SPMD L1 | 动态相对多task耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        static_us = row["static_spmd_historical_reference"]["median_us"]
        static_text = "—" if static_us is None else f"{static_us:.2f}"
        lines.append(f"| {row['history']//1024}K / B{row['batch']} | {row['native_historical_external_profiler']['median_us']:.2f} | "
                     f"{row['dynamic_pto_full_l1_us']:.2f} | {static_text} | {row['single_spmd_full_l1_us']:.2f} | "
                     f"{row['spmd_time_change_percent']:+.2f}% |")
    lines += [
        "", "耗时变化正值为更慢，负值为更快。七档均未达到至少降低30%的目标。", "",
        "## 轮次与版本说明", "",
        "- B16正式版本最近一轮为多task869.78、动态889.40；静态最新已有值781.94来自另一轮，其同轮多task是880.06。不能把781.94当成与869.78同轮。",
        "- B16最新4辅助核实验候选为947.16，同轮多task880.06，退步7.62%，未采用；候选源码与全部验证保存在projection_branch_evidence。",
        "- 历史B16多task905.08及静态807.16保留。905.08→880.06涉及编译环境和采样流程差异，未隔离具体成因；核对见projection_branch_evidence/MULTITASK_905_VS_880_AUDIT.json。",
        "- B28 Native为10月18日优化测量，已开静态编译、SuperKernel及NZ29投影权重，30次外部profiling中位1701.75；不带profiler的Event20次中位1653.08，未混入表格。",
        "- B28多task/动态列已有5次同轮采样，动态版本为9d5327e5；66882634新增的QA小档分支不作用于T168。最新依赖环境没有重测B28。",
        "- B28静态既有任务在导入阶段失败，无性能数据；按只用已有静态结果的要求留空，不补跑静态。其他三项齐全，无需补测。",
        "- 其余Native来自10月12日已有前后两轮各30次合并中位数，不能当作10月18日同轮环境。其他静态列来自各档已验收历史记录。", "",
        "## 逐档PTO配对来源", "",
    ]
    for row in rows:
        lines.append(f"- {row['history']//1024}K/B{row['batch']}: `{row['pto_pair_task']}`；[{Path(row['pto_pair_source']).parent.name}]({row['pto_pair_source']})。")
    lines += ["", "同名JSON保存全部样本、环境、源码版本、静态同轮基线和原始结果路径；本文件不把跨轮绝对差解释为优化收益。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(stem.with_suffix(".md"))
    for row in rows:
        print(row["history"], row["batch"], row["native_historical_external_profiler"]["median_us"],
              row["dynamic_pto_full_l1_us"], row["static_spmd_historical_reference"]["median_us"],
              row["single_spmd_full_l1_us"], round(row["spmd_time_change_percent"], 2))


if __name__ == "__main__":
    main()
