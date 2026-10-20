# SPDX-License-Identifier: Apache-2.0
"""汇总完整五窗配对结果；不挑最快窗口，不把跨轮差异或PMU时长拼成收益。"""

import json
from pathlib import Path
from statistics import median


here = Path(__file__).resolve().parent
rows = []
for task in json.loads((here / "FULL_TASKS.json").read_text()):
    report = Path(task["output"]) / "result.json"
    data = json.loads(report.read_text())
    assert data["passed"] and data["stage"] == "complete"
    assert data["full_csa_validated"] and data["performance_measured"]
    matrix = data["source_matrix"]
    assert matrix["passed"] and matrix["same_process"] and matrix["same_root_addresses"]
    assert all(value["passed"] for value in matrix["protocol"].values())
    stats = matrix["statistics_us"]
    samples = {key: value["samples_us"] for key, value in stats.items()}
    assert all(len(value) == 5 for value in samples.values())
    assert all(median(value["samples_us"]) == value["median_us"] for value in stats.values())
    reference = stats["spmd_reference"]["median_us"]
    candidate = stats["spmd_candidate"]["median_us"]
    multi = stats["dynamic_pto"]["median_us"]
    rows.append({
        "batch": task["batch"], "history": task["history"],
        "multi_task_l1_us": multi, "reference_spmd_l1_us": reference,
        "candidate_spmd_l1_us": candidate, "net_saved_us": reference - candidate,
        "time_increase_percent": 100 * (candidate - reference) / reference,
        "improvement_percent_user_formula": 100 * (reference - candidate) / candidate,
        "paired_saved_us": [a - b for a, b in zip(
            samples["spmd_reference"], samples["spmd_candidate"]
        )],
        "samples_us": samples,
        "full_precision_passed": True, "graph_and_protocol_passed": True,
        "task": task["task"], "result": str(report),
    })
result = {
    "scope": "128K/S6，正式W8A8/NZ；同进程同工具链同输入五窗交替完整L1 incore首尾",
    "rows": rows,
    "at_least_one_focus_saves_20us": any(row["net_saved_us"] >= 20 for row in rows),
    "formal_adopted": False,
    "expand_shapes": False,
    "decision": "两重点档完整中位耗时均回退，停止本O_B放置候选；保留新版SDMA功能修复",
    "not_proven": "未采集此候选阶段PMU，不能把退步唯一归因于某种带宽或缓存竞争",
    "goal_complete": False,
}
assert not result["at_least_one_focus_saves_20us"]
(here / "OB_PERFORMANCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
lines = [
    "# 新版 SDMA 的 O_B 后段候选：完整对照结果",
    "",
    "同一新工具链、SDMA-enabled context、冻结公共源码和正式 W8A8/NZ 输入。",
    "下表为完整 CSA 的 L1 最早 incore 开始至最晚结束，五个交替窗口的中位数，单位 μs。",
    "",
    "| 场景 | 多 task L1 | 原单 SPMD | O_B预取候选 | 净省 | 耗时增加 |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| 128K/B{row['batch']}/S6 | {row['multi_task_l1_us']:.2f} | "
        f"{row['reference_spmd_l1_us']:.2f} | {row['candidate_spmd_l1_us']:.2f} | "
        f"{row['net_saved_us']:.2f} | {row['time_increase_percent']:.2f}% |"
    )
lines += [
    "",
    "两档本轮A/B输入的完整精度、Top-K、cache/state、保护区、图重放与同步代际通过。",
    "本次仅验证S6与本轮合成历史，不代表S1/S5或全部动态metadata边界通过。",
    "两档均未达到净省20μs的候选门槛，不扩其余档位、不采用正式入口。",
    "新版SDMA组件的完整MIX功能修复继续保留；不能把该功能修复当作性能优化已生效。",
    "",
    "B16五个配对窗口全部变慢；B28四个变慢、一个变快，完整中位数仍变慢。",
    "不删除B28参照的较慢窗口，也不以最快窗口或不同轮次基线作有利比较。",
    "本次未额外采候选阶段PMU，不能声称已证明退步由某一种争用独占导致。",
    "停止本放置策略的参数扫描，回到目标的其他有证据方向；不据此否定所有SDMA用途。",
    "",
    "全部窗口、任务号及结果路径见 `OB_PERFORMANCE.json`。动态范围继续保留，",
    "候选不进入正式算子；新版独立工具链与完整原型留作后续有新证据时复用。",
]
(here / "OB_PERFORMANCE.md").write_text("\n".join(lines) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
