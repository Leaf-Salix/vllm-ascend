# SPDX-License-Identifier: Apache-2.0
"""汇总八档已有退避对照，保留每轮自己的参考，禁止跨轮混算收益。

三列分别展示50/100、100/150、150/200的同轮配对。重复试验独立列在
附表，不挑更快的一次覆盖首轮。未测和未结束的场景明确保留为空。
"""

import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFESTS = (
    "WAIT_BACKOFF_TASKS.json",
    "WAIT_BACKOFF_EXPANSION_TASKS.json",
    "WAIT_BACKOFF_200_TASKS.json",
    "WAIT_BACKOFF_200_FOLLOWUP_TASKS.json",
    "WAIT_BACKOFF_B1_TASKS.json",
)
PAIRS = ((50, 100), (100, 150), (150, 200))


def collect():
    records = []
    for name in MANIFESTS:
        manifest = HERE / name
        if not manifest.exists():
            continue
        for task in json.loads(manifest.read_text()):
            if task.get("query_tokens", 6) != 6:
                continue
            directory = REPO / task["output"]
            result_file = directory / "result.json"
            audit_file = directory / "WAIT_BACKOFF_COMPARISON.json"
            row = {
                key: task[key] for key in (
                    "task", "history", "batch", "candidate_ticks", "reference_ticks", "output"
                )
            }
            row.update(complete=False, passed=False)
            if result_file.exists() and audit_file.exists():
                result = json.loads(result_file.read_text())
                audit = json.loads(audit_file.read_text())
                row.update(
                    complete=True,
                    passed=bool(result["passed"] and audit["passed"]),
                    full_csa_validated=result["full_csa_validated"],
                    graph_replay_validated=result["graph_replay_validated"],
                )
                measurements = {}
                for role in ("candidate", "reference"):
                    stats = audit["statistics_us"]["spmd_" + role]
                    assert len(stats["samples_us"]) == 5, task["task"]
                    assert statistics.median(stats["samples_us"]) == stats["median_us"], task["task"]
                    measurements[str(task[role + "_ticks"])] = stats
                row["measurements_by_tick"] = measurements
            records.append(row)
    return records


def pair_label(row, pair):
    data = row["measurements_by_tick"]
    return " / ".join(f"{data[str(tick)]['median_us']:.2f}" for tick in pair)


def main():
    cases = json.loads((HERE / "PERFORMANCE_CASES.json").read_text())["cases"]
    records = collect()
    primary, repeated = {}, []
    for row in records:
        pair = tuple(sorted((row["candidate_ticks"], row["reference_ticks"])))
        key = (row["history"], row["batch"], pair)
        if key in primary:
            repeated.append(row)
        else:
            primary[key] = row
    report = dict(
        metric="完整L1 incore首尾、含内部等待；单位us，每侧5窗中位数",
        pairs=[list(pair) for pair in PAIRS],
        cases=cases, records=records,
        cross_round_speedup_allowed=False,
        b1_specific_policy_authorized=True,
    )
    (HERE / "WAIT_BACKOFF_ALL_PERFORMANCE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 八档50/100/150/200 tick实测",
        "",
        "单位μs，完整L1 incore首尾（包含内部等待），五窗中位数。"
        "每个斜线左右是一轮直接配对，按列头顺序读取。",
        "不同列属于不同测量轮次，100和150可以出现不同耗时；"
        "不能跨列相减或拼出单轮收益。原七档不重跑，只补新增B1。",
        "",
        "| 档位 | 50 / 100 tick | 100 / 150 tick | 150 / 200 tick |",
        "|---|---:|---:|---:|",
    ]
    for case in cases:
        label = f"{'128K' if case['history'] == 131072 else '8K'}/B{case['batch']}"
        cells = [label]
        for pair in PAIRS:
            row = primary.get((case["history"], case["batch"], pair))
            cells.append(
                "未测" if row is None else pair_label(row, pair) if row["passed"]
                else "未通过" if row["complete"] else "等待结果"
            )
        lines.append("| " + " | ".join(cells) + " |")
    if repeated:
        lines += ["", "## 独立复核（不覆盖首轮）", ""]
        for row in repeated:
            pair = tuple(sorted((row["candidate_ticks"], row["reference_ticks"])))
            if row["passed"]:
                lines.append(
                    f"- {row['history'] // 1024}K/B{row['batch']}，"
                    f"{pair[0]}/{pair[1]} tick：{pair_label(row, pair)} μs；"
                    f"任务`{row['task']}`。"
                )
    lines += [
        "",
        "各任务的完整数值、Top-K/score、状态、保护区、图重放、同步代际与"
        "真实CCEC输入审计保存在同名JSON所列output目录。",
        "B1使用真实B1/S6。此前B1/S1检查只作为功能覆盖，未代替本轮性能。",
    ]
    (HERE / "WAIT_BACKOFF_ALL_PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
