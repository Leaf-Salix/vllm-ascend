# SPDX-License-Identifier: Apache-2.0
"""汇总150对100或200对150的七档配对实测，不跨轮拼接基线。"""
import argparse
import json
from pathlib import Path
import statistics


CASES = ((131072, 4), (131072, 8), (131072, 16), (131072, 24),
         (131072, 28), (8192, 16), (8192, 32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ticks", type=int, choices=(150, 200), default=150)
    args = parser.parse_args()
    candidate_ticks = args.candidate_ticks
    reference_ticks = 100 if candidate_ticks == 150 else 150
    report_prefix = "WAIT_BACKOFF_SEVEN" if candidate_ticks == 150 else "WAIT_BACKOFF_200"
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    tasks = {}
    task_files = (("WAIT_BACKOFF_TASKS.json", "WAIT_BACKOFF_EXPANSION_TASKS.json")
                  if candidate_ticks == 150 else ("WAIT_BACKOFF_200_TASKS.json",))
    for name in task_files:
        for task in json.loads((here / name).read_text()):
            if task["candidate_ticks"] != candidate_ticks:
                continue
            if task["reference_ticks"] != reference_ticks:
                raise AssertionError("对照间隔与本轮定义不符")
            key = task["history"], task["batch"]
            if key in tasks:
                raise ValueError(f"同一档存在重复结果，不允许择优取样：{key}")
            tasks[key] = task
    if set(tasks) != set(CASES):
        raise ValueError("七档定义或任务清单不完整")
    rows = []
    for history, batch in CASES:
        task = tasks[history, batch]
        out = repo / task["output"]
        row = dict(history=history, batch=batch, task=task["task"],
                   output=task["output"], candidate_ticks=candidate_ticks, reference_ticks=reference_ticks,
                   complete=False, passed=False)
        result_file, audit_file = out / "result.json", out / "WAIT_BACKOFF_COMPARISON.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            row["stage"] = result.get("stage")
            row["error"] = result.get("error")
            if audit_file.exists():
                audit = json.loads(audit_file.read_text())
                row["complete"] = True
                row["passed"] = bool(result["passed"] and audit["passed"])
                row["full_csa_validated"] = result["full_csa_validated"]
                row["graph_replay_validated"] = result["graph_replay_validated"]
                row["actual_source_audit"] = audit["passed"]
                stats = audit["statistics_us"]
                ref, cand = stats["spmd_reference"], stats["spmd_candidate"]
                if len(ref["samples_us"]) != 5 or len(cand["samples_us"]) != 5:
                    raise AssertionError("每侧须完整保留五窗")
                deltas = [round(x - y, 4) for x, y in
                          zip(ref["samples_us"], cand["samples_us"], strict=True)]
                row.update(
                    reference_us=ref["median_us"],
                    candidate_us=cand["median_us"],
                    saved_us=round(ref["median_us"] - cand["median_us"], 4),
                    speedup_pct=(ref["median_us"] - cand["median_us"]) / cand["median_us"] * 100,
                    reference_samples_us=ref["samples_us"],
                    candidate_samples_us=cand["samples_us"],
                    paired_saved_us=deltas,
                    paired_saved_median_us=statistics.median(deltas),
                    faster_windows=sum(x > 0 for x in deltas),
                )
        rows.append(row)
    report = dict(
        complete=all(row["complete"] for row in rows),
        passed=all(row["passed"] for row in rows),
        criterion="用户已取消20us门槛；按完整L1正收益、配对分布、功能和受影响范围决定",
        metric="完整L1 incore最早开始到最晚结束，含内部等待；每侧五窗中位数",
        source=f"正式动态SPMD冻结source_reference，两侧只变等待helper的{reference_ticks}/{candidate_ticks}常量",
        sampling=("B16/B28沿用本轮已有重点档；其余五档补测；每行均有同轮同卡100参考"
                  if candidate_ticks == 150 else "七档200均与同轮同卡150直接配对；不复用上一轮150耗时"),
        speedup_formula=f"({reference_ticks}tick耗时-{candidate_ticks}tick耗时)/{candidate_ticks}tick耗时*100",
        formal_default_changed=False,
        rows=rows,
    )
    if report["passed"]:
        report["median_faster_cases"] = sum(row["saved_us"] > 0 for row in rows)
        report["all_seven_medians_faster"] = all(row["saved_us"] > 0 for row in rows)
        report["all_paired_windows_faster"] = all(row["faster_windows"] == 5 for row in rows)
    (here / (report_prefix + "_PERFORMANCE.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        f"# 七档{candidate_ticks} tick与{reference_ticks} tick退避对照",
        "",
        f"用户已取消20μs门槛。单位μs，完整L1首尾五窗中位数；"
        f"每行均有独立同轮{reference_ticks}参考。正净省表示{candidate_ticks}更快。",
        f"提升比例按用户指定：({reference_ticks}耗时−{candidate_ticks}耗时)÷{candidate_ticks}耗时。",
        "",
        f"| 档位 | {reference_ticks} tick | {candidate_ticks} tick | 净省 | 提升 | 更快窗口 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = f"{'128K' if row['history'] == 131072 else '8K'}/B{row['batch']}"
        if row["passed"]:
            lines.append(
                f"| {name} | {row['reference_us']:.2f} | {row['candidate_us']:.2f} | "
                f"{row['saved_us']:+.2f} | {row['speedup_pct']:+.2f}% | {row['faster_windows']}/5 |"
            )
        else:
            lines.append(f"| {name} | — | 待完成或失败 | — | — | — |")
    lines.extend([
        "",
        ("128K/B16、B28沿用此前两组150配对，未重复测试。其余五档为后补实测。"
         if candidate_ticks == 150 else "200的七档均与本轮150直接比较，不使用此前150结果作为同轮基线。"),
        "逐窗数据、任务ID、完整数值/状态/保护/图检查与实际生成源码审计见同名JSON及各任务目录。",
        "表中小幅中位数差异不等于所有窗口都改善；正式默认是否修改另行记录。",
    ])
    (here / (report_prefix + "_PERFORMANCE.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
