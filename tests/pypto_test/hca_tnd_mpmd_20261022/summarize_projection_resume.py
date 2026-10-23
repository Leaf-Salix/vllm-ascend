# SPDX-License-Identifier: Apache-2.0
"""归档10月23日投影分档实验；只分析已有文件/队列日志，不重新运行测试。"""

import json
import shutil
import subprocess
from pathlib import Path

from summarize_native_output import collect
from summarize_q_projection import stage_statistics

HERE = Path(__file__).resolve().parent
CPU_RUNS = (
    "q_tall_direct_cpu_v1", "oa_tall_cpu_v1", "q_tall_policy_cpu_v1",
    "q_tall_policy_cpu_v2", "ob_tall_cpu_v1", "projection_policy_cpu_v4",
    "dynamic_return_alias_v1", "dynamic_return_alias_v2", "dynamic_return_alias_v3",
)


def main():
    manifest = HERE / "Q_RESUME_TASKS.json"
    tasks = json.loads(manifest.read_text())
    paths = set()
    for task in tasks:
        task["status"] = subprocess.check_output(
            ["task-submit", "--status", task["task_id"]], text=True,
        ).strip()
        if not task["status"].startswith("completed"):
            raise RuntimeError(f"任务尚未结束，不能最终归档：{task['task_id']} {task['status']}")
        log = HERE / "evidence" / (task["name"] + ".log")
        log.write_text(subprocess.check_output(["task-submit", "--log", task["task_id"]], text=True))
        task["log"] = str(log.relative_to(HERE))
        run = HERE / "runs" / task["name"]
        paths.add(run)
        command = task["command"]
        paths.add(Path(command[command.index("--experiment") + 1]))
        shutil.copy2(run / "result.json", HERE / "evidence" / (task["name"] + "_result.json"))
    manifest.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    rows = collect("Q_RESUME_TASKS.json")
    for row in rows:
        run = HERE / "runs" / row["name"]
        raw = json.loads((run / "result.json").read_text())
        row["selected_policy"] = raw.get("selected_policy")
        row["reference_policy"] = raw.get("reference_policy", "default")
        row["archived_result"] = "evidence/" + row["name"] + "_result.json"
        if "paired_l1" in row:
            pair = row["paired_l1"]
            row["faster_pairs"] = sum(a > b for a, b in zip(
                pair["control"]["samples_us"], pair["candidate"]["samples_us"]))
            row["stages"] = {side: stage_statistics(run, side) for side in ("control", "candidate")}
    cpu = {}
    for name in CPU_RUNS:
        path = HERE / "runs" / name
        paths.add(path)
        result = path / "result.json"
        cpu[name] = json.loads(result.read_text())
        shutil.copy2(result, HERE / "evidence" / (name + "_result.json"))
    for name in ("q_resume_base_v1", "q_tall_policy_v1", "q_tall_policy_v2", "q_tall_policy_v3",
                 "ob_tall_v1", "projection_policy_v4"):
        paths.add(HERE / "experiments" / name)
    artifacts = []
    for path in sorted(paths):
        files = [p for p in path.rglob("*") if p.is_file()]
        artifacts.append({"path": str(path), "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
    (HERE / "PROJECTION_RESUME_ARTIFACTS.json").write_text(json.dumps({
        "policy": "权重/二进制/原始泳道仅本地保留；路径/数量/大小归档，无hash扫描",
        "local": artifacts,
    }, ensure_ascii=False, indent=2) + "\n")
    report = {
        "metric": "完整HCA直接执行的L1 incore首尾；同任务配对；无ACLGraph replay",
        "policy": "NZ且Native已有Host max_seq_lens>=131072、129<=T<=192选择Q_B/O_B M192/N128",
        "not_adopted": ["O_A M192/N128及Q_B+O_A组合", "64K启用大T投影策略"],
        "native_strict_storage_pass": False,
        "cpu": cpu, "rows": rows,
    }
    (HERE / "PROJECTION_RESUME_SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 2026-10-23 HCA投影区间优化实测", "",
             "同任务、同SDK的直接调用L1首尾，单位μs；提升比例按`control/candidate−1`。",
             "每行只比较自己的控制组，不跨时间窗相减，不把不同候选的收益相加。",
             "Native严格storage原差异仍保留，不宣称Native全面精度或30%目标通过。", ""]
    for title, selected in (
        ("最终Q_B/O_B独立入口相对本轮原版", [r for r in rows if r["name"].startswith("projection_final_")]),
        ("其余试验，包括负结果", [r for r in rows if not r["name"].startswith("projection_final_")]),
    ):
        lines += ["## " + title, "", "| 实验 | 场景 | 对照 | 候选 | 性能提升 | 更快配对 |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in selected:
            if "paired_l1" not in row:
                continue
            pair = row["paired_l1"]
            label = f"{max(row['histories']) // 1024}K/B{len(row['queries'])}/T{sum(row['queries'])}"
            lines.append(f"| {row['name']} | {label} | {pair['control']['median_us']:.2f} "
                         f"| {pair['candidate']['median_us']:.2f} | {row['speedup_pct_user_definition']:+.2f}% "
                         f"| {row['faster_pairs']}/{len(pair['candidate']['samples_us'])} |")
        lines.append("")
    lines += ["`ob_tall_increment_*`的控制已含Q_B优化；其余控制为本轮原版。", "",
              "## 服务与功能", ""]
    for row in rows:
        if "service_pass" in row:
            lines.append(f"- `{row['name']}`：policy={row['selected_policy']}，服务={row['service_pass']}，"
                         f"控制输出精确={row['control_output_exact']}，控制storage精确={row['control_storage_exact']}。")
    lines += ["", "全部原始样本、阶段统计、任务ID、精度与本地大产物路径见同名JSON及ARTIFACTS清单。",
              "完整编译失败及未能复现相同错误的最小探针也保留；未改共享SDK。", ""]
    (HERE / "PROJECTION_RESUME_SUMMARY.md").write_text("\n".join(lines))
    print(HERE / "PROJECTION_RESUME_SUMMARY.md")


if __name__ == "__main__":
    main()
