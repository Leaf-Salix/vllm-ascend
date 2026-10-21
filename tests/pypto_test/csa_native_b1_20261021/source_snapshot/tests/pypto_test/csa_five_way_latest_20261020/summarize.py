# SPDX-License-Identifier: Apache-2.0
"""仅汇总本次成功的七档实测，检查优化生效证据；不读取旧表、不挑最快样本。"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
CASES = [(131072, b) for b in (4, 8, 16, 24, 28)] + [(8192, b) for b in (16, 32)]


def load_phase(case, phase):
    filename = "report.json" if phase in {"native", "multitask_outer"} else "result.json"
    reports = sorted(case.glob(f"{phase}_v*/{filename}"), key=lambda p: p.stat().st_mtime_ns)
    if not reports:
        raise ValueError(f"尚无 {phase} 报告")
    path = reports[-1]
    report = json.loads(path.read_text())
    if phase in {"native", "multitask_outer"}:
        assert report["status"] == "MEASURED", (path, report.get("error", report["status"]))
    else:
        assert report["stage"] == "complete" and report["passed"], (path, report.get("error", report["stage"]))
    return path, report


def outer(path, report, native):
    assert set(report["native_weight_formats"].values()) == {29}
    options = report["backend_options"]
    assert options["force_eager"] is False and options["static_kernel_compile"] is True
    assert options["inplace_pass"] and not options["clone_input"] and not options["clone_output"]
    assert report["torch_compile_options"] == {"dynamic": False, "fullgraph": True}
    assert report["measurement"]["graph_count"] == 1 and report["measurement"]["host_updated_nodes"] == 0
    assert all(v["status"] == "PASS" for v in report["timing"]["eager_comparison"].values())
    profile = report["device_profile"]
    assert profile["available"] and profile["span_includes_all_streams"] and profile["replays"] == 30
    samples = profile["span_us_all"]
    assert len(samples) == 30 and len(set(profile["kernels_per_replay"])) == 1
    with Path(profile["raw_csv"]).open() as stream:
        names = Counter(row["Name"] for row in csv.DictReader(stream))
    evidence = {"report": str(path), "nz_formats": report["native_weight_formats"],
                "compiler": report["compiler"], "graph": report["measurement"],
                "kernel_counts": dict(names), "profile": profile,
                "samples_us": samples, "median_us": median(samples),
                "min_us": min(samples), "max_us": max(samples)}
    if native:
        compiler = report["compiler"]
        assert compiler["wrapper_compiled"] and compiler["installed_static_packages"] > 0
        assert compiler["static_compile_results"] and all(compiler["static_compile_results"])
        assert compiler["static_super_flags"] and all(compiler["static_super_flags"])
        assert options["super_kernel_optimize"] and report["multistream"]["dsa_overlap"]
        calls = report["super_kernel_graph_calls"]
        assert calls and all(x["status"] == "RETURNED_SUCCESSFULLY" for x in calls)
        # 同时检查真实trace中的融合静态kernel和NZ矩阵乘，避免只看开关。
        evidence["static_super_kernels"] = {k: v for k, v in names.items() if k.startswith("sk_") and "static_kernel" in k}
        evidence["nz_matmul_kernels"] = {k: v for k, v in names.items() if "MatmulWeightNz" in k}
        assert evidence["static_super_kernels"] and evidence["nz_matmul_kernels"]
        evidence["super_kernel_graph_calls"] = calls
    else:
        assert report["implementation_package"] == "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"
        assert report["compiler"]["pto_dispatch_calls"] > 0
        evidence["aicpu_kernel_count"] = sum(v for k, v in names.items() if k.startswith("simpler_aicpu_kernel_exec"))
        evidence["aicore_kernel_count"] = names["aicore_kernel_mode_0"]
        assert evidence["aicpu_kernel_count"] == 30 and evidence["aicore_kernel_count"] == 30
    return median(samples), evidence


def incore(path, report, static):
    assert set(report["nz_weight_formats"].values()) == {29}
    if static:
        assert report["graph_replay_passed"] and report["compile"]["cached100_verified"]
        name = "static_spmd"
    else:
        assert report["full_csa_validated"] and report["graph_replay_validated"]
        assert report["performance_measured"] and report["post_l1_protocol"]
        assert report["wait_backoff"] and not report["phase_clock"]
        name = "single_spmd"
    l1 = report["l1"]
    rows = l1["windows"]
    assert len(rows) == 10
    for start in range(0, 10, 2):
        assert {row["side"] for row in rows[start:start+2]} == {"dynamic_pto", name}
        expected = ["dynamic_pto", name] if (start // 2) % 2 == 0 else [name, "dynamic_pto"]
        assert [row["side"] for row in rows[start:start+2]] == expected, "实际采样必须AB/BA交替"
    for row in rows:
        assert row["clock_freq_hz"] == 50000000 and row["cores"] == 72
        if row["side"] == name:
            assert row["task_count"] == 1
    samples = {side: [r["incore_span_us"] for r in rows if r["side"] == side]
               for side in ("dynamic_pto", name)}
    assert all(len(values) == 5 for values in samples.values())
    return {side: median(values) for side, values in samples.items()}, {
        "report": str(path), "nz_formats": report["nz_weight_formats"], "windows": rows,
        "samples_us": samples, "statistics_us": l1["statistics_us"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    results, evidence, errors = [], {}, {}
    for history, batch in CASES:
        key = f"h{history}_b{batch}"
        case = HERE / "results" / key
        row = {"history": history, "batch": batch, "query_tokens": 6}
        proof = {}
        for phase in ("native", "multitask_outer", "l1", "static"):
            if phase == "static" and (history, batch) == (131072, 28):
                row["static_l1_us"] = None
                row["static_note"] = "没有已验收的128K/B28静态实现，未运行、不修复、不以邻档替代"
                continue
            try:
                path, report = load_phase(case, phase)
                assert report["history"] == history and report["batch"] == batch
                if phase in {"native", "multitask_outer"}:
                    value, proof[phase] = outer(path, report, native=phase == "native")
                    row[phase + "_us"] = value
                else:
                    values, proof[phase] = incore(path, report, static=phase == "static")
                    if phase == "l1":
                        row.update(multitask_l1_us=values["dynamic_pto"], dynamic_l1_us=values["single_spmd"])
                    else:
                        row["static_l1_us"] = values["static_spmd"]
                        row["static_paired_multitask_l1_us"] = values["dynamic_pto"]
            except (AssertionError, ValueError, KeyError, FileNotFoundError) as exc:
                errors[f"{key}/{phase}"] = str(exc)
        for baseline, field in (("native_us", "gain_over_native_pct"), ("multitask_outer_us", "gain_over_multitask_outer_pct")):
            if baseline in row and "dynamic_l1_us" in row:
                row[field] = (row[baseline] - row["dynamic_l1_us"]) / row["dynamic_l1_us"] * 100
        results.append(row)
        evidence[key] = proof
    output = {"measured_date": "2026-10-20", "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "all_requested_available_cases_passed": not errors, "errors_or_pending": errors,
              "formula": "(baseline_us - dynamic_l1_us) / dynamic_l1_us * 100",
              "scope": "同日同卡分阶段；Native/PTO外层全流首尾，SPMD/多task L1为incore首尾；口径不同",
              "dynamic_implementation": "当前正式13259f92算法；未采用Key多ND等负例",
              "rows": results, "evidence": evidence}
    (HERE / "PERFORMANCE.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    if errors and not args.partial:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))
    columns = ("native_us", "multitask_outer_us", "multitask_l1_us", "static_l1_us", "dynamic_l1_us",
               "gain_over_native_pct", "gain_over_multitask_outer_pct")
    lines = ["# CSA七档五口径最新实测（2026-10-20）", "", "单位μs；全部S6、NZ2，表中为各窗口中位数。", "",
             "| 档位 | Native | 多task含AICPU | 多task L1 | 静态SPMD | 动态SPMD | 较Native提升 | 较多task含AICPU提升 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append("—" if value is None else f"{value:.2f}" + ("%" if col.startswith("gain_") else ""))
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | " + " | ".join(cells) + " |")
    lines += ["", "提升=(基准−动态SPMD)/动态SPMD。Native和多task含AICPU列是Profiler外层全流设备跨度；",
              "后三列是L1 incore最早开始至最晚结束，保留内部等待。比例属于上述两种口径的对照，",
              "不是同口径的整服务吞吐提升。128K/B28静态版无可用实现，保持空值。", "",
              "Native逐档核验静态编译/安装、SuperKernel成功返回、真实sk_*static_kernel和",
              "MatmulWeightNz trace、唯一ACLGraph重放及四张核心投影格式29。全部原始样本、",
              "实际设备路径、编译返回值和trace信息见PERFORMANCE.json的evidence。", ""]
    if errors:
        lines += ["本文件为阶段性结果，尚有未完成/未通过项目：", "", *[f"- {k}: {v}" for k, v in errors.items()], ""]
    (HERE / "PERFORMANCE.md").write_text("\n".join(lines))
    print("\n".join(lines[:15]))
    print("未完成/未通过",len(errors))


if __name__ == "__main__":
    main()
