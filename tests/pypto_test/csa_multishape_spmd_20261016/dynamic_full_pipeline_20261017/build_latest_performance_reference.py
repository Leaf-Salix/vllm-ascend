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


def outer_multitask_records():
    """外部跨度明确含AICPU/AICore；旧六档不与当前L1相减冒充纯调度开销。"""
    historical = HERE.parents[4] / ".worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/hc_pre_layer_integration_20261012/summary.json"
    report = read(historical)
    records = {}
    for case in report["cases"]:
        key = (case["history"], case["batch"])
        if case["kind"] != "csa" or key == (8192, 24):
            continue
        baseline = case["passes"]["baseline"]
        corrected_path = Path(baseline["report"]).parent / "device_profile_layer_only.json"
        corrected = read(corrected_path)
        samples = corrected["span_us_all"]
        assert len(samples) == 30 and statistics.median(samples) == corrected["span_us"] == baseline["p50_us"]
        records[key] = {
            "status": "MEASURED", "median_us": statistics.median(samples),
            "samples_us": samples, "count": len(samples), "date": "2026-10-12",
            "source_commit": "a00afa9f671803228280c2f200373b062ff36cf5",
            "source": str(corrected_path), "raw_csv": corrected["raw_csv"],
            "kernels_per_replay": corrected["kernels_per_replay"],
            "kernel_roles": ["simpler_aicpu_kernel_exec", "aicore_kernel_mode_0"],
            "metric": "全部设备流最早CSA kernel开始至最晚结束，包含AICPU/AICore完整调度，剔除图外reset/poison",
            "same_run_as_current_l1": False,
        }
    b28_path = HERE / "multitask_outer_profile_20261018/h131072_b28_v1/summary.json"
    if b28_path.exists():
        value = read(b28_path)
        assert value["status"] == "MEASURED" and value["count"] == 30
        assert statistics.median(value["samples_us"]) == value["median_us"]
        records[(131072, 28)] = {
            **value, "date": "2026-10-18", "source": str(b28_path),
            "task": "task_20261018_122158_10688028513",
            "source_commit": "6e23806d", "same_run_as_current_l1": False,
        }
    else:
        records[(131072, 28)] = {"status": "PENDING", "median_us": None,
                                 "task": "task_20261018_122158_10688028513"}
    return records


def main():
    historical = read(HERE / "FOUR_WAY_PERFORMANCE_REFERENCE_20261018.json")
    rows = deepcopy(historical["rows"])
    outer_records = outer_multitask_records()
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
        row["multitask_outer_profile"] = outer_records[(row["history"], row["batch"])]
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
        "new_measurements": outer_records[(131072, 28)]["status"] == "MEASURED",
        "new_measurement_scope": "只补128K/B28的原分task外部完整跨度；其余四列及六档外层值使用已有数据",
        "metric_native": historical["native_metric"], "metric_pto": historical["pto_metric"],
        "comparability": "每行PTO多task L1和动态SPMD L1同轮；Native、外层多task及静态是独立轮次；外层与L1不可跨轮相减为纯调度开销",
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
        "正式第4层W8A8、S=6、完整CSA含HC_pre/HC_post，单位μs，均为中位数。仅补缺失的B28多task外层profiling，其余使用已有结果。", "",
        "动态列选当前正式采用策略的最近实测，不从实验候选中挑最小值。每行多task L1与动态SPMD L1同轮；外层多task和静态列取独立轮次。", "",
        "Native和新增多task外层列是外部profiler完整设备kernel跨度，包含AICPU调度；L1三列是完整incore首尾，排除外围首尾。两种口径不能直接算加速比。", "",
        "| 历史 / Batch | Native外部profiling | PTO多task含AICPU外层 | PTO多task L1 | 静态SPMD L1（历史） | 动态单SPMD L1 | 动态相对多task L1耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        static_us = row["static_spmd_historical_reference"]["median_us"]
        static_text = "—" if static_us is None else f"{static_us:.2f}"
        outer = row["multitask_outer_profile"]["median_us"]
        outer_text = "待补测结束" if outer is None else f"{outer:.2f}"
        lines.append(f"| {row['history']//1024}K / B{row['batch']} | {row['native_historical_external_profiler']['median_us']:.2f} | "
                     f"{outer_text} | {row['dynamic_pto_full_l1_us']:.2f} | {static_text} | {row['single_spmd_full_l1_us']:.2f} | "
                     f"{row['spmd_time_change_percent']:+.2f}% |")
    lines += [
        "", "耗时变化正值为更慢，负值为更快。七档均未达到至少降低30%的目标。", "",
        "## 轮次与版本说明", "",
        "- B16正式版本最近一轮为多task869.78、动态889.40；静态最新已有值781.94来自另一轮，其同轮多task是880.06。不能把781.94当成与869.78同轮。",
        "- B16最新4辅助核实验候选为947.16，同轮多task880.06，退步7.62%，未采用；候选源码与全部验证保存在projection_branch_evidence。",
        "- 历史B16多task905.08及静态807.16保留。905.08→880.06涉及编译环境和采样流程差异，未隔离具体成因；核对见projection_branch_evidence/MULTITASK_905_VS_880_AUDIT.json。",
        "- B28 Native为10月18日优化测量，已开静态编译、SuperKernel及NZ29投影权重，30次外部profiling中位1701.75；不带profiler的Event20次中位1653.08，未混入表格。",
        "- B28多task/动态列已有5次同轮采样，动态版本为9d5327e5；66882634新增的QA小档分支不作用于T168。最新依赖环境没有重测B28。",
        "- B28静态既有任务在导入阶段失败，无性能数据；按只用已有静态结果的要求留空，不补跑静态。Native/多task L1/动态L1已有数据；新增含AICPU外层列独立补测。",
        "- 其余Native来自10月12日已有前后两轮各30次合并中位数，不能当作10月18日同轮环境。其他静态列来自各档已验收历史记录。", "",
        "- 含AICPU的多task外层列：六档沿用10月12日a00afa9f原分task，每次包含一个simpler_aicpu_kernel_exec和一个aicore_kernel_mode_0，取两者整体跨度。B28为当前源码补测。",
        "- 这些外层记录与表内L1并非同轮同版本，不能用两列相减得出纯AICPU开销。L1 incore首尾本来就包含中间调度造成的空隙；新增列补的是完整外部可见边界，不累加重叠时间。", "",
        "## 逐档PTO配对来源", "",
    ]
    for row in rows:
        lines.append(f"- {row['history']//1024}K/B{row['batch']}: `{row['pto_pair_task']}`；[{Path(row['pto_pair_source']).parent.name}]({row['pto_pair_source']})。")
    lines += ["", "同名JSON保存全部样本、环境、源码版本、静态同轮基线和原始结果路径；本文件不把跨轮绝对差解释为优化收益。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(stem.with_suffix(".md"))
    for row in rows:
        print(row["history"], row["batch"], row["native_historical_external_profiler"]["median_us"],
              row["multitask_outer_profile"]["median_us"], row["dynamic_pto_full_l1_us"], row["static_spmd_historical_reference"]["median_us"],
              row["single_spmd_full_l1_us"], round(row["spmd_time_change_percent"], 2))


if __name__ == "__main__":
    main()
