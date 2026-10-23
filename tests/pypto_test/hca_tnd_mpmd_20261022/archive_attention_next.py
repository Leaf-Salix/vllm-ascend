# SPDX-License-Identifier: Apache-2.0
"""归档累计max/UB候选与混长参考定位；仅读取已有任务，不发起设备执行。"""

import concurrent.futures
import json
from pathlib import Path
import re
import shutil
import subprocess

from summarize_attention import attention_stages
from summarize_native_output import all_passed

HERE = Path(__file__).resolve().parent
MANIFESTS = ("ATTENTION_ONLINE_TASKS.json", "ATTENTION_BOUNDARY_TASKS.json", "ATTENTION_PAIR_TASKS.json",
             "ATTENTION_GATHER_TASKS.json", "ATTENTION_GATHER_DISPATCH_TASKS.json", "ATTENTION_DIRECT_KV_TASKS.json")
CPU_RUNS = ("attention_online_cpu_v1", "attention_online_n64_cpu_v1",
            "attention_pair_plan_cpu_v1", "attention_pair_plan_cpu_v2", "attention_pair_plan_cpu_v3",
            *(f"attention_pair512_cpu_v{i}" for i in range(1, 7)), "attention_gather_mask_cpu_v1",
            "attention_gather_mask_policy_cpu_v2", "attention_reset_alpha_h32_cpu_v1",
            *(f"attention_gather_dispatch_cpu_v{i}" for i in range(1, 4)),
            *(f"attention_direct_cmp_kv_cpu_v{i}" for i in range(1, 4)), "dead_tile_init_cpu_v1")


def status(task):
    return subprocess.check_output(["task-submit", "--status", task["task_id"]], text=True).strip()


def main():
    rows, local = [], set()
    for name in MANIFESTS:
        path = HERE / name
        tasks = json.loads(path.read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            statuses = list(pool.map(status, tasks))
        for task, state in zip(tasks, statuses):
            task["status"] = state
            row = {"name": task["name"], "task_id": task["task_id"], "status": state,
                   "scope": task.get("scope", task.get("control_note", task.get("note")))}
            rows.append(row)
            if not state.startswith("completed"):
                continue
            run = HERE / "runs" / task["name"]
            local.add(run)
            log = HERE / "evidence" / (task["name"] + ".log")
            log.write_text(subprocess.check_output(["task-submit", "--log", task["task_id"]], text=True))
            task["log"] = str(log.relative_to(HERE))
            for source in (run / "result.json", run / "precision_probe.json", run / "native_capture_steps.jsonl"):
                if source.exists():
                    target = HERE / "evidence" / (task["name"] + "_" + source.name)
                    shutil.copy2(source, target)
            source = run / "result.json"
            if not source.exists():
                row["result"] = "没有结果文件；退出码及原始日志保留"
                continue
            result = json.loads(source.read_text())
            for key in ("passed", "stage", "aclgraph_replay", "shared_toolchain", "traceback", "cases",
                        "native_active_reference", "precision_probe", "compressor_probe"):
                if key in result:
                    row[key] = result[key]
            if "pto_native" in result:
                row["native_output"] = result["pto_native"]["output"]
                row["native_strict_storage_pass"] = all_passed(
                    {k: v for k, v in result["pto_native"].items() if k != "output"})
            if "pto_reference" in result:
                row["control_storage_exact"] = all_passed(
                    {k: v for k, v in result["pto_reference"].items() if k != "output"})
            if "paired_l1" in result:
                pair = result["paired_l1"]
                row["paired_l1"] = {side: {key: value[key] for key in ("samples_us", "median_us")}
                                    for side, value in pair.items()}
                a, b = (pair[side]["median_us"] for side in ("control", "candidate"))
                row["net_saved_us"] = a - b
                row["speedup_pct"] = (a / b - 1) * 100
                row["faster_pairs"] = sum(y < x for x, y in zip(pair["control"]["samples_us"],
                                                               pair["candidate"]["samples_us"]))
                row["attention_stage"] = {side: attention_stages(
                    run, side, ("hca_unified_attention_aic", "hca_unified_attention_aiv", "_pair_cube", "_pair_vector",
                                "hca_pair_plan", "hca_cmp_work_gather", "hca_cmp_page_plan"))
                    for side in ("control", "candidate")}
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    cpu = {}
    for name in CPU_RUNS:
        run = HERE / "runs" / name
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            trace = result.get("traceback", "")
            if len(trace) > 100000:
                # SSA诊断会重复整份模型IR；原42MB结果本地保留，Git只存错误摘要及路径/大小。
                result.pop("traceback")
                result["traceback_excerpt"] = trace[:3000]
                result["diagnostic_messages"] = sorted(set(re.findall(r"Message: ([^\n]+)", trace)))
                result["local_full_result"] = {"path": str(run / "result.json"),
                                                "bytes": (run / "result.json").stat().st_size}
            cpu[name] = result
            (HERE / "evidence" / (name + "_result.json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            local.add(run)
    for pattern in ("attention_online*", "attention_boundary_diag*", "attention_pair*", "attention_gather*",
                    "attention_reset_alpha*", "attention_direct*"):
        local.update((HERE / "experiments").glob(pattern))
    artifacts = []
    for path in sorted(local):
        files = [p for p in path.rglob("*") if p.is_file()]
        artifacts.append({"path": str(path), "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
    report = {"production_head": "37bbd0f3c12d32966695ad45e7059557d1fc03ad",
              "production_changed": False, "metric": "完整HCA eager L1 incore首尾；无ACLGraph",
              "reference_correction": "旧混长padding Native Q仅15行而cu_q尾为24，参考无效；v6统一真实范围后剩1项输出超差",
              "adoption": "全部新候选未采用；核外gather分档及直接KV也未获益；正式算子仍为37bbd0f3c的已验证策略",
              "work_state": "本轮已收口，按用户再次要求在归档提交后暂停；未完成30%目标，未push",
              "component_issue": "https://github.com/hw-native-sys/pypto/issues/3136",
              "rows": rows, "cpu": cpu, "local_artifacts_no_hash": artifacts}
    (HERE / "ATTENTION_NEXT_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# HCA attention后续候选与混长参考核对", "",
             "共享SDK csa-shared-20261022-logical-codegen；以下均为已有结果，不新增设备测试。",
             "本轮已收口，用户再次要求结束后停下；归档提交后暂停。所有新候选均未采用，正式算子仍为37bbd0f3c的区间策略。", "",
             "| 候选/场景 | 控制μs | 候选μs | 性能提升 | 更快配对 | Native输出 |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if "paired_l1" not in row:
            continue
        pair = row["paired_l1"]
        lines.append(f"| {row['name']} | {pair['control']['median_us']:.2f} | {pair['candidate']['median_us']:.2f} "
                     f"| {row['speedup_pct']:+.2f}% | {row['faster_pairs']}/{len(pair['candidate']['samples_us'])} "
                     f"| {row['native_output']['status']} |")
    lines += ["", "N64/H8的控制是累计max候选，不能把该增量当作相对正式算子的收益。",
              "全部严格Native cache/state检查尚未通过，任务正常退出或输出PASS不等于整层验收。", "",
              "## 尚未结束的设备任务", ""]
    pending = [row for row in rows if not row['status'].startswith('completed')]
    for row in pending:
        if not row['status'].startswith('completed'):
            lines.append(f"- `{row['name']}`：{row['status']}，`{row['task_id']}`。")
    if not pending:
        lines.append("无。已提交设备任务全部终态；不再启动下一轮，按用户要求暂停。")
    lines += ["",
              "## 混长padding参考的纠正", "",
              "旧Native输入Q=[15,64,512]而cu_q包含padding至24，产生无效参考或MTE错误。",
              "v5只裁metadata导致RoPE维度不匹配；v6同时裁metadata、hidden、position和output，",
              "PTO仍测试原T24/B8容量。v6不再出现MTE，输出剩1个超差[12,0,17]，属于history=0/Q1。",
              "给Native Q或Native cache单独替换时输出均通过；两者都用PTO时复现该项。",
              "Native attention输入PTO输出尾部逐bit一致，定位剩余误差在前置Q/KV的组合而非O尾部。",
              "源码复核：公共HCA mHC已采用(y0+y1)+(y2+y3)，也有BF16舍入，不能声称这两项遗漏。", "",
              "CPU编译、输入快照审计和每项失败详情见JSON及evidence/。快照和编译产物只本地保留，",
              "清单记录路径与大小，不做hash；不把失配参考的旧失败改写为PASS。", ""]
    (HERE / "ATTENTION_NEXT_RESULTS.md").write_text("\n".join(lines))
    print(HERE / "ATTENTION_NEXT_RESULTS.md")


if __name__ == "__main__":
    main()
