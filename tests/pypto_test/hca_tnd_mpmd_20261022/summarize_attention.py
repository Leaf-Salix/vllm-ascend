# SPDX-License-Identifier: Apache-2.0
"""归档attention候选的已存在结果；不编译、不提交NPU任务、不计算hash。

把完整层L1首尾和attention各核耗时分开记录，避免将平均核耗时当作端到端收益。
排队中任务可以保留pending；失败、未通过严格Native storage也必须进入报告。
"""

import collections
import json
import shutil
import statistics
import subprocess
from pathlib import Path

from summarize_native_output import all_passed

HERE = Path(__file__).resolve().parent
MANIFESTS = ("ATTENTION_PV_TASKS.json", "ATTENTION_Q_L0_TASKS.json", "ATTENTION_MACRO_TASKS.json",
             "ATTENTION_POLICY_TASKS.json")
CPU_RUNS = (
    "attention_pv_db_cpu_v1", "attention_pv_dual_cpu_v2", "attention_q_l0_cpu_v1",
    "attention_macro512_cpu_v1", "attention_macro512_cpu_v2", "attention_macro512_cpu_v3",
    "attention_macro512_h32_cpu_v1", "attention_macro512_qk2_cpu_v1",
    "fifo_offset_codegen_v1", "fifo_offset_codegen_v2", "fifo_offset_codegen_v3",
    "attention_policy_cpu_v1", "attention_policy_cpu_v2", "attention_policy_cpu_v3", "attention_policy_cpu_v4",
    "reduction_ring_insert_v1", "reduction_ring_insert_v2",
)


def attention_stages(run, side, prefixes=("hca_unified_attention_aic", "hca_unified_attention_aiv")):
    """Worker View中的每次实际执行只计一次；不把Task View重复加进来。"""
    samples = collections.defaultdict(list)
    for path in sorted(run.glob(f"l1_*_{side}/swimlane.json")):
        events = [e for e in json.loads(path.read_text())["traceEvents"]
                  if e.get("ph") == "X" and e.get("pid") == 4]
        begin = min(e["ts"] for e in events)
        for prefix in prefixes:
            selected = [e for e in events if e["name"].startswith(prefix)]
            if selected:
                samples[prefix].append({
                    "trace": str(path), "begin_us": min(e["ts"] for e in selected) - begin,
                    "end_us": max(e["ts"] + e["dur"] for e in selected) - begin,
                    "mean_kernel_duration_us": statistics.mean(e["args"]["kernel-duration-us"] for e in selected),
                    "max_kernel_duration_us": max(e["args"]["kernel-duration-us"] for e in selected),
                    "records": len(selected),
                })
    return {name: {"medians": {key: statistics.median(r[key] for r in rows)
                              for key in ("begin_us", "end_us", "mean_kernel_duration_us",
                                          "max_kernel_duration_us", "records")}, "samples": rows}
            for name, rows in samples.items()}


def main():
    rows, artifacts = [], set()
    for manifest in MANIFESTS:
        path = HERE / manifest
        if not path.exists():
            continue
        tasks = json.loads(path.read_text())
        for task in tasks:
            status = subprocess.check_output(["task-submit", "--status", task["task_id"]], text=True).strip()
            task["status"] = status
            row = {"name": task["name"], "task_id": task["task_id"], "status": status,
                   "manifest": manifest}
            row["control_note"] = task.get("control_note", task.get("note"))
            rows.append(row)
            run = HERE / "runs" / task["name"]
            if not status.startswith("completed"):
                continue
            log = HERE / "evidence" / (task["name"] + ".log")
            log.write_text(subprocess.check_output(["task-submit", "--log", task["task_id"]], text=True))
            task["log"] = str(log.relative_to(HERE))
            artifacts.add(run)
            source = run / "result.json"
            if not source.exists():
                row["result"] = "未生成result；失败以队列退出码及日志为准"
                continue
            result = json.loads(source.read_text())
            archive = HERE / "evidence" / (task["name"] + "_result.json")
            shutil.copy2(source, archive)
            row["archived_result"] = str(archive.relative_to(HERE))
            row["shared_toolchain"] = result.get("shared_toolchain")
            row["stage"] = result.get("stage", result.get("status"))
            row["passed_all_strict_checks"] = result.get("passed")
            for key in ("query_lengths", "history_lengths", "token_capacity", "attention_math_transition",
                        "execution_mode", "aclgraph_replay", "selected_policy", "reference_policy", "traceback"):
                if key in result:
                    row[key] = result[key]
            if "cases" in result:
                row["independent_probe_cases"] = result["cases"]
            if "pto_native" in result:
                row["native_output"] = result["pto_native"]["output"]
                row["strict_native_storage_pass"] = all_passed({k: v for k, v in result["pto_native"].items()
                                                               if k != "output"})
                row["repeat_exact"] = all_passed(result["repeat_eager"])
                row["guards_pass"] = all(all_passed(result[key]) for key in
                                         ("native_guards", "pto_guards", "repeat_guards"))
            if "pto_reference" in result:
                row["control_output_exact"] = all_passed(result["pto_reference"]["output"])
                row["control_storage_exact"] = all_passed({k: v for k, v in result["pto_reference"].items()
                                                          if k != "output"})
            if "reference_native" in result:
                row["control_native_output"] = result["reference_native"]["output"]
            if "service_validation" in result:
                row["service_pass"] = result["service_validation"]["passed"]
                row["service_validation"] = result["service_validation"]
            if "paired_l1" in result:
                pair = result["paired_l1"]
                row["paired_l1"] = {side: {k: v[k] for k in ("samples_us", "median_us")}
                                    for side, v in pair.items()}
                a, b = (pair[side]["median_us"] for side in ("control", "candidate"))
                row["latency_reduction_us"] = a - b
                row["speedup_pct_user_definition"] = (a / b - 1) * 100
                row["faster_pairs"] = sum(y < x for x, y in zip(pair["control"]["samples_us"],
                                                               pair["candidate"]["samples_us"]))
                row["stages"] = {side: attention_stages(run, side) for side in ("control", "candidate")}
                row["post_l1_guards_pass"] = all_passed(result["post_l1_guards"])
                row["post_l1_repeat_exact"] = all_passed(result["post_l1_eager"])
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    cpu = {}
    for name in CPU_RUNS:
        source = HERE / "runs" / name / "result.json"
        if source.exists():
            artifacts.add(source.parent)
            cpu[name] = json.loads(source.read_text())
            shutil.copy2(source, HERE / "evidence" / (name + "_result.json"))
    for pattern in ("attention_*",):
        artifacts.update((HERE / "experiments").glob(pattern))
    sizes = []
    for path in sorted(artifacts):
        files = [f for f in path.rglob("*") if f.is_file()]
        sizes.append({"path": str(path), "files": len(files), "bytes": sum(f.stat().st_size for f in files)})
    (HERE / "ATTENTION_ARTIFACTS.json").write_text(json.dumps({
        "policy": "原始泳道、编译产物和冻结工作副本本地保留；只统计路径/大小，不做hash校验",
        "local": sizes,
    }, ensure_ascii=False, indent=2) + "\n")
    report = {
        "metric": "完整HCA直接调用L1 incore首尾；同任务配对；无ACLGraph",
        "production_attention_changed": True,
        "policy": "NZ；Native已有Host max_seq_lens>=131072且T>=33启用K512；T129..192叠加既有Q_B/O_B分档；其他K128",
        "acceptance": "用户允许按区间选择后接入；K512改变舍入。Native严格storage和混长C128边界遗留不能冒充通过",
        "issue": "https://github.com/hw-native-sys/pypto/issues/3130",
        "cpu": cpu, "rows": rows,
    }
    (HERE / "ATTENTION_SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# HCA TND attention：2026-10-23 核内流水候选", "",
             "以同任务双侧完整层L1首尾作比较，单位μs；性能提升=`control/candidate−1`。",
             "正式入口按区间选择K128/K512；大块候选改变舍入，局部/整层检查不代表整网token验收。",
             "NZ且Native已有Host max_seq_lens≥131072、T≥33时用K512；其他仍用K128。",
             "T129～192再叠加已验证的Q_B/O_B投影档。所有边界/实际长度继续读取device Q/L。",
             "Native严格storage既有差异保留。没有ACLGraph回放，没有新增Native性能测量。", "",
             "| 试验 | 控制 | 候选 | 性能提升 | 更快配对 | Native输出 | 控制storage精确 |",
             "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        if "paired_l1" not in row:
            continue
        pair = row["paired_l1"]
        lines.append(f"| {row['name']} | {pair['control']['median_us']:.2f} "
                     f"| {pair['candidate']['median_us']:.2f} | {row['speedup_pct_user_definition']:+.2f}% "
                     f"| {row['faster_pairs']}/{len(pair['candidate']['samples_us'])} "
                     f"| {row['native_output']['status']} | {row['control_storage_exact']} |")
    lines += ["", "## 未列性能的任务", ""]
    for row in rows:
        if "paired_l1" not in row:
            lines.append(f"- `{row['name']}`：{row['status']}，{row.get('stage', '无结果文件')}。")
    lines += ["", "完整样本、attention AIC/AIV各核耗时、功能结果、失败CPU编译和任务身份见同名JSON。",
              "源码来源、设计差异与不采用项见`ATTENTION_NATIVE_PIPELINE_REVIEW.md`。", ""]
    (HERE / "ATTENTION_SUMMARY.md").write_text("\n".join(lines))
    print(HERE / "ATTENTION_SUMMARY.md")


if __name__ == "__main__":
    main()
