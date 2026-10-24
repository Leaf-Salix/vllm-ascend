# SPDX-License-Identifier: Apache-2.0
"""只读汇总HCA八档三列，核对优化开关、有效样本和原始记录；不提交设备任务。"""

import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

from pto_interval import operator_interval

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def passed(value):
    if isinstance(value, list):
        return all(passed(v) for v in value)
    if isinstance(value, dict):
        if "status" in value:
            return value["status"] == "PASS"
        return all(passed(v) for v in value.values())
    return True


def make_cell(samples, *, task, raw, kind):
    if len(samples) < 10 or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError(f"{task['key']}没有至少10个完整有效窗口")
    return {"status": "MEASURED", "median_us": statistics.median(samples), "samples_us": samples,
            "valid_samples": len(samples), "task_id": task["task_id"], "task_key": task["key"],
            "source_report": str(raw), "kind": kind, "collection": "new_measurement"}


def main():
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    manifest = read(HERE / "manifest.json")
    cases = read(HERE / "cases.json")["cases"]
    tasks = read(HERE / "tasks.json")
    columns = ["Native", "HCA-MPMD-AICPU", "HCA-MPMD-L1"]
    cells = {c["id"]: {col: {"status": "待采集"} for col in columns} for c in cases}
    attempts, artifacts, correctness = [], [], {}
    for task in tasks:
        status = subprocess.check_output(["task-submit", "--status", task["task_id"]], text=True).strip()
        row = dict(task, queue_status=status)
        path = Path(task["output"]) / ("report.json" if task["side"] == "native" else "result.json")
        if not path.exists():
            attempts.append(row)
            continue
        data = read(path)
        target = evidence / f"{task['key']}.json"
        shutil.copy2(path, target)
        row["archived_report"] = str(target.relative_to(HERE))
        row["stage"] = data.get("stage", data.get("status"))
        row["error"] = data.get("error", data.get("traceback"))
        if status.startswith("completed"):
            log = subprocess.check_output(["task-submit", "--log", task["task_id"]], text=True)
            (evidence / f"{task['key']}.log").write_text(log)
        attempts.append(row)
        if task["side"] == "cpu":
            continue
        case_id = task["case_id"]
        if status != "completed (exit=0)":
            if status.startswith("completed"):
                for col in (["Native"] if task["side"] == "native" else columns[1:]):
                    cells[case_id][col] = {"status": "FAIL", "task_id": task["task_id"]}
            continue
        if task["side"] == "native":
            compiler = data["compiler"]
            if (data["status"] != "MEASURED" or not all(compiler["static_compile_results"])
                    or not all(compiler["static_super_flags"]) or not compiler["installed_static_packages"]
                    or not data["super_kernel_graph_calls"]
                    or any(c["status"] != "RETURNED_SUCCESSFULLY" for c in data["super_kernel_graph_calls"])
                    or set(data["native_weight_formats"].values()) != {29}
                    or not data["multistream"]["dsa_overlap"]
                    or not passed(data["initial_guards"])
                    or not passed(data["timing"]["eager_comparison"])
                    or not passed(data["timing"]["guards"])):
                raise ValueError(f"{task['key']}未实际启用全套优化或数值/保护区检查失败")
            profile = data["device_profile"]
            cells[case_id]["Native"] = make_cell(profile["span_us_all"], task=task, raw=path,
                                                   kind="全部相关device stream的完整图执行首尾")
            cells[case_id]["Native"]["path_evidence"] = {
                "compiler": compiler, "superkernel": data["super_kernel_graph_calls"],
                "weight_formats": data["native_weight_formats"], "multistream": data["multistream"],
                "measurement": data["measurement"], "backend_options": data["backend_options"],
                "streams": profile["stream_ids_per_replay"], "raw_csv": profile["raw_csv"],
            }
            traces = list((Path(task["output"]) / "profile").glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
            for p in [Path(profile["raw_csv"]), *traces]:
                artifacts.append({"task_id": task["task_id"], "path": str(p), "bytes": p.stat().st_size})
        else:
            perf = data["performance_spec"]
            if (data["stage"] != "completed" or not data["preserves_reference"]
                    or data["pto_native"]["output"]["status"] != "PASS"
                    or not passed(data["repeat_eager"]) or not passed(data["repeat_guards"])
                    or not perf["service_validation"]["passed"] or perf["aclgraph_replay"]
                    or perf["shared_manifest"] != manifest["shared_toolchain"]
                    or set(perf["native_weight_formats"].values()) != {29}):
                raise ValueError(f"{task['key']}实现/公共SDK/已有精度门禁不匹配")
            outer = perf["aicpu"]
            for w in outer["windows"]:
                if "AI_CPU" not in w["task_types"] or "MIX_AIC" not in w["task_types"]:
                    raise ValueError("外部窗口没有同时记录AICPU与AICore")
            if len(outer["window_checks"]) != len(outer["samples_us"]):
                raise ValueError("缺少逐窗状态检查")
            for check in [*outer["window_checks"], *(w["checks"] for w in perf["l1"]["windows"])]:
                if not all(check.values()):
                    raise ValueError("采样窗口数值或保护区无效")
            operator = operator_interval(outer)
            cells[case_id]["HCA-MPMD-AICPU"] = make_cell(
                operator["samples_us"], task=task, raw=path, kind="完整PTO调用全部device活动首尾，含AICPU、AICore、DMA与等待")
            cells[case_id]["HCA-MPMD-AICPU"]["boundary_evidence"] = operator
            cells[case_id]["HCA-MPMD-L1"] = make_cell(
                perf["l1"]["samples_us"], task=task, raw=path, kind="整个PTO整层所有incore最早开始至最晚结束")
            correctness[case_id] = {"native_output": data["pto_native"]["output"],
                                    "native_cache_state": {k: v for k, v in data["pto_native"].items() if k != "output"},
                                    "unchanged_vs_frozen_operator": data["preserves_reference"],
                                    "all_native_strict_checks_pass": data["passed"],
                                    "selected_policy": data["selected_policy"],
                                    "device": data["device"], "environment": {k: perf[k] for k in
                                      ("versions", "cann", "custom_opp", "device_name", "native_weight_formats")}}
            paths = [Path(outer["trace"])]
            for w in perf["l1"]["windows"]:
                paths += [Path(w["raw"]), Path(w["trace"])]
            for p in paths:
                artifacts.append({"task_id": task["task_id"], "path": str(p), "bytes": p.stat().st_size})
    rows = []
    for case in cases:
        cell = cells[case["id"]]
        native, l1 = cell["Native"], cell["HCA-MPMD-L1"]
        ratio = None
        if native["status"] == l1["status"] == "MEASURED":
            ratio = 100 * (native["median_us"] - l1["median_us"]) / l1["median_us"]
        rows.append({"case": case, "metrics": cell, "native_over_l1_percent": ratio})
    result = {"manifest": manifest, "rows": rows, "attempts": attempts,
              "correctness": correctness, "large_artifacts": artifacts,
              "limitation": "保留Native cache/state既有严格阈值差异；不是完整模型或连续轨迹验收"}
    (HERE / "PERFORMANCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (HERE / "ARTIFACTS.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n")
    lines = ["# HCA八档性能：Native全优化，PTO直接调用", "", "单位μs，中位数。Native 30窗，PTO每列10窗。", "",
             "| 场景 | Native | HCA-MPMD-AICPU | HCA-MPMD-L1 | (Native − L1) / L1 |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        case = row["case"]
        label = f"{case['history_label']//1024}K/B{case['active_requests']}/" + (
            "混长T56" if case["id"] == "C08" else "S6")
        values = [f"{row['metrics'][c]['median_us']:.2f}" if row['metrics'][c]['status'] == "MEASURED"
                  else row['metrics'][c]['status'] for c in columns]
        ratio = row["native_over_l1_percent"]
        lines.append("| " + " | ".join([label, *values, "—" if ratio is None else f"{ratio:.2f}%"]) + " |")
    lines += ["", "Native已核实静态编译、SuperKernel、NZ29、多stream与ACLGraph。PTO不使用ACLGraph。",
              "含AICPU列为完整PTO算子的全部device活动首尾，不是AICPU自身耗时；额外采到的服务前导及派发间隙另存诊断。",
              "Native输出原容差通过；cache/state原有严格阈值差异保留，全部PTO结果对冻结正式算子逐bit一致。",
              "", "全部样本、任务号、数值检查、失败尝试与原始trace位置见 [PERFORMANCE.json](PERFORMANCE.json)。"]
    (HERE / "PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
