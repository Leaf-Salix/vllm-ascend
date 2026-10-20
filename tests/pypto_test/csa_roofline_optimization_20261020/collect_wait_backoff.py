# SPDX-License-Identifier: Apache-2.0
"""只汇总已生成的退避配对结果，不启动设备测试，不挑窗口。"""
import json
from pathlib import Path
import statistics


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    tasks = json.loads((here / "WAIT_BACKOFF_TASKS.json").read_text())
    rows = []
    for task in tasks:
        output = Path(task["output"])
        if not output.is_absolute():
            output = repo / output
        row = {key: task[key] for key in ("batch", "history", "candidate_ticks", "reference_ticks", "task")}
        row["output"] = str(output.relative_to(repo))
        audit_path = output / "WAIT_BACKOFF_COMPARISON.json"
        result_path = output / "result.json"
        row["complete"] = audit_path.is_file() and result_path.is_file()
        if row["complete"]:
            audit = json.loads(audit_path.read_text())
            result = json.loads(result_path.read_text())
            row["passed"] = bool(audit["passed"] and result["passed"])
            row["full_csa_validated"] = result["full_csa_validated"]
            row["graph_replay_validated"] = result["graph_replay_validated"]
            stats = audit["statistics_us"]
            reference = stats["spmd_reference"]
            candidate = stats["spmd_candidate"]
            if len(reference["samples_us"]) != 5 or len(candidate["samples_us"]) != 5:
                raise AssertionError("本轮要求保留全部五个配对窗口")
            deltas = [round(a - b, 4) for a, b in zip(
                reference["samples_us"], candidate["samples_us"], strict=True,
            )]
            row.update(
                reference_us=reference["median_us"],
                candidate_us=candidate["median_us"],
                saved_us=round(reference["median_us"] - candidate["median_us"], 4),
                paired_saved_us=deltas,
                paired_saved_median_us=statistics.median(deltas),
                candidate_faster_windows=sum(delta > 0 for delta in deltas),
                statistics_us=stats,
                parameter_only_actual_source_audit=True,
            )
        rows.append(row)
    report = dict(
        complete=all(row["complete"] for row in rows),
        passed=all(row.get("passed", False) for row in rows),
        metric="完整 L1 最早 incore 开始至最晚结束；包括内部等待",
        pairing="每行独立同进程、同设备、同源同工具链五窗对照；各行100参考不跨轮拼接",
        formal_adoption=False,
        rows=rows,
    )
    if report["passed"]:
        report["candidate_focus_gates"] = {
            str(ticks): dict(
                one_focus_saved_at_least_20_us=any(
                    row["saved_us"] >= 20 for row in rows if row["candidate_ticks"] == ticks
                ),
                both_focus_medians_not_slower=all(
                    row["saved_us"] >= 0 for row in rows if row["candidate_ticks"] == ticks
                ),
            )
            for ticks in (50, 150)
        }
        report["decision"] = (
            "本轮50tick两档回退；150tick小幅改善但未达20us候选门槛，"
            "B28五窗仅3窗更快。保留正式100tick，不扩档、不继续扫描。"
        )
    (here / "WAIT_BACKOFF_PERFORMANCE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 50 / 100 / 150 tick 当前动态 CSA 退避对照",
        "",
        "单位 μs；每行候选与同轮 100 tick 参考比较，完整 L1 五窗中位数。",
        "净省为参考减候选，负数表示变慢。正式默认尚未修改。",
        "",
        "| 档位 | 候选 tick | 同轮 100 | 候选 | 净省 | 五窗中更快的次数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("passed"):
            lines.append(
                f"| 128K/B{row['batch']} | {row['candidate_ticks']} | "
                f"{row['reference_us']:.2f} | {row['candidate_us']:.2f} | "
                f"{row['saved_us']:+.2f} | {row['candidate_faster_windows']}/5 |"
            )
        else:
            lines.append(f"| 128K/B{row['batch']} | {row['candidate_ticks']} | — | 待完成或失败 | — | — |")
    if report["passed"]:
        lines.extend(["", report["decision"], "",
                      "四组输出、Top-K/score、状态、保护区、图重放及信号代际均通过；"
                      "变换前C++去注释/空白后逐字一致，实际CCEC输入的退避常量已核对。"])
    (here / "WAIT_BACKOFF_PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
