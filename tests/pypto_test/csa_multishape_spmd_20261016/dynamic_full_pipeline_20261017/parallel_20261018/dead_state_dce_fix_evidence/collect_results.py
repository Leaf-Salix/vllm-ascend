#!/usr/bin/env python3
"""汇总已提交任务；只读原始测量，不重复运行算子，也不采用中途报告。"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = {(131072, b) for b in (4, 8, 16, 24, 28)} | {(8192, b) for b in (16, 32)}


def read_result(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # 工作进程尚未写出或正在更新报告，不能把临时内容当最终结果。
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    registry = HERE / "DEVICE_TASKS.json"
    tasks = json.loads(registry.read_text())
    rows = []
    for task in tasks:
        tid = task["stdout"].strip()
        if not task.get("last_status", "").startswith("completed"):
            task["last_status"] = subprocess.run(
                ["task-submit", "--status", tid], check=True, text=True, capture_output=True
            ).stdout.strip()
        output = Path(task["output"])
        result = read_result(output / "result.json")
        matrix = read_result(output / "source_matrix/result.json")
        row = dict(history=task["history"], batch=task["batch"], task=tid,
                   status=task["last_status"], stage=result.get("stage"),
                   passed=task["last_status"] == "completed (exit=0)" and
                   result.get("passed") is True and matrix.get("passed") is True,
                   target_case=(task["history"], task["batch"]) in TARGETS,
                   full_csa_validated=result.get("full_csa_validated", False),
                   result=str(output / "result.json"), matrix=str(output / "source_matrix/result.json"))
        if row["passed"]:
            assert matrix["same_process"] and matrix["same_root_addresses"]
            row["statistics_us"] = matrix["statistics_us"]
            row["candidate_vs_reference"] = matrix["candidate_vs_reference"]
            row["comparison"] = matrix["comparison"]
            row["policy"] = result["policy"]
        rows.append(row)
    registry.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    rows.sort(key=lambda r: (-r["history"], r["batch"]))
    report = dict(complete=all(r["passed"] for r in rows), timing="完整L1首尾、同进程五窗中位数",
                  compiler="独立PyPTO内存复用树+死循环状态DCE修复；双方相同",
                  rows=rows, target_met=sum(r["passed"] and r["target_case"] and
                      r["comparison"]["spmd_candidate"]["target_met"] for r in rows))
    (HERE / "DEVICE_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 区间候选与原路径：修复后编译器的设备结果", "",
             "单位 μs；完整 L1，原分 task / 原动态 / 本地区间候选同进程五窗中位数。", "",
             "| 历史/B | 用途 | 原多 task | 原动态 | 区间候选 | 相对原动态降低 | 状态 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        name = f"{row['history']//1024}K/B{row['batch']}"
        if row["passed"]:
            stats = row["statistics_us"]
            values = [stats[n]["median_us"] for n in ("dynamic_pto", "spmd_reference", "spmd_candidate")]
            data = " | ".join(f"{v:.2f}" for v in values)
            gain = f"{row['candidate_vs_reference']['latency_reduction_pct']:+.3f}%"
            state = "PASS"
        else:
            data, gain, state = "— | — | —", "—", f"{row['status']} / {row['stage']}"
        lines.append(f"| {name} | {'七档' if row['target_case'] else '区间边界'} | {data} | {gain} | {state} |")
    lines += ["", "负值为回退，完整五窗原始数值保存在 DEVICE_RESULTS.json 与每个 source_matrix/result.json。",
              "不跨编译器归因收益，不把 CPU/阶段插桩当设备精度或完整性能。",
              f"本候选达到相对原多 task 降低30%的回归档数：{report['target_met']}/7。", ""]
    (HERE / "DEVICE_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if args.require_complete and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
