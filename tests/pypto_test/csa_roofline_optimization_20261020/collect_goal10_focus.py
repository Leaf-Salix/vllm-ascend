# SPDX-License-Identifier: Apache-2.0
"""汇总新目标重点档的已完成测量，独立工作区与同址诊断分开呈现。

不覆盖原始数据、不跨轮合并或选最快值。反向复核显式交换角色解释。
旧runner的30%字段属于历史协议，本报告按用户新公式重新计算10%目标。
"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXPERIMENTS = (
    ("B1初始A/A", "GOAL10_B1_BASELINE_TASK.json", False),
    ("成对压缩直写", "PAIR_DIRECT_FULL_TASKS.json", False),
    ("B28反向复核", "PAIR_DIRECT_REVERSE_B28_TASK.json", True),
    ("B1/4K分叶", "B1_LEAF4K_TASK.json", False),
    ("同址诊断", "SHARED_SCRATCH_DIAGNOSTIC_TASKS.json", False),
    ("B1/UB根同址", "B1_ROOT_RESIDENT_TASK.json", False),
    ("B1/UB根独立", "B1_ROOT_PRIVATE_TASK.json", False),
    ("WS列池/150", "SCORE_POOL_GOAL10_TASKS.json", False),
    ("B1/原半叶", "B1_LEGACY_HALF_FULL_TASK.json", False),
    ("B1/半叶分数留UB", "B1_HALF_SCORE_RESIDENT_FULL_TASK.json", False),
    ("B1/正式源确认", "B1_FORMAL_CONFIRM_TASK.json", False),
    ("连续scale读取", "SCALE_FOCUS_TASKS.json", False),
    ("Score归属/未激活诊断", "SCORE_OWNER_FULL_TASK.json", False),
    ("Score归属/实际重排", "SCORE_OWNER_ACTIVE_TASK.json", False),
    ("Score归属/同址开关", "SCORE_OWNER_ABLATION_TASK.json", False),
    ("Score归属/反向正常区", "SCORE_OWNER_REVERSE_TASK.json", True),
    ("Score归属/B24回退", "SCORE_OWNER_BOUNDARY_TASK.json", False),
    ("Score归属/均衡快速证明", "SCORE_OWNER_UNIFORM_TASKS.json", False),
    ("WS列池与scale区间组合", "SCORE_STREAM_TASK.json", False),
    ("WS列池与scale反向", "SCORE_STREAM_REVERSE_TASK.json", True),
    ("B1/O_B M16", "B1_OB16_TASK.json", False),
    ("两区间集成确认", "FOCUS_COMBINED_TASKS.json", False),
    ("B24关闭区间反向", "FOCUS_COMBINED_B24_REVERSE_TASK.json", True),
)


def main():
    rows = []
    for label, manifest, reversed_roles in EXPERIMENTS:
        path = HERE / manifest
        if not path.exists():
            continue
        tasks = json.loads(path.read_text())
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            output = REPO / task["output"]
            result_file = output / "result.json"
            row = dict(label=label, task=task["task"], batch=task["batch"],
                       output=task["output"], complete=False,
                       shared_scratch=task.get("shared_scratch", False),
                       source_freeze=task.get("source_freeze", "SHARED_SOURCE_GOAL10.json"),
                       adopted=False)
            row["source_candidate"] = task.get("candidate")
            row["source_reference"] = task.get("reference")
            if label == "Score归属/未激活诊断":
                row["adoption_blocker"] = "布尔true编码为-1，==1条件未成立；本轮没有重排，不能归因为重排收益。"
                row["active_mapping"] = False
            if label == "Score归属/同址开关":
                row["reference_note"] = "同接口同容量身份映射诊断；不是原正式源码，不能当作正式改善量。"
            if label in ("B1/4K分叶", "B1/UB根同址", "B1/UB根独立"):
                row["adoption_blocker"] = (
                    "H139264/S5 全同分边界失败；2560个Top-K索引及57510个输出元素不同。"
                    "普通S6性能仍是实测，但不能认定达到可采用的10%目标。"
                )
            adoption_path = HERE / "B1_HALF_SCORE_ADOPTION.json"
            if label in ("B1/半叶分数留UB", "B1/正式源确认") and adoption_path.exists():
                row["adopted"] = json.loads(adoption_path.read_text()).get("adopted", False)
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if (result.get("passed") and
                        (output / "CURRENT_WAIT_POLICY_AUDIT.json").exists()):
                    stats = result["source_matrix"]["statistics_us"]
                    proposal_role = "spmd_reference" if reversed_roles else "spmd_candidate"
                    control_role = "spmd_candidate" if reversed_roles else "spmd_reference"
                    proposal, control = stats[proposal_role], stats[control_role]
                    baseline = stats["dynamic_pto"]["median_us"]
                    latency = proposal["median_us"]
                    improvement = (baseline - latency) / latency * 100
                    row.update(
                        complete=True, passed=True, statistics_us=stats,
                        proposal_role=proposal_role, control_role=control_role,
                        multitask_us=baseline, control_spmd_us=control["median_us"],
                        proposal_spmd_us=latency,
                        improvement_vs_multitask_pct=improvement,
                        saved_vs_control_us=control["median_us"] - latency,
                        faster_than_control_windows=sum(a > b for a, b in
                            zip(control["samples_us"], proposal["samples_us"], strict=True)),
                        additional_us_for_10pct=max(latency - baseline / 1.1, 0),
                        median_reaches_10pct=improvement >= 10,
                    )
            rows.append(row)
    report = dict(
        target_improvement_pct=10,
        formula="(多task完整L1-候选完整L1)/候选完整L1",
        scope="128K/S6，单位us，各角色五窗中位数；独立工作区与同址诊断不合并",
        source_freeze="SHARED_SOURCE_GOAL10.json",
        goal_complete=False, rows=rows,
    )
    (HERE / "GOAL10_FOCUS_PERFORMANCE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 新10%目标：重点档阶段记录", "",
        report["scope"] + "。提升比例为" + report["formula"] + "。",
        "A/A没有算法候选；同址诊断用于排查偏移，不能直接代替独立工作区的采用验证。", "",
        "**4K分叶及UB根旧候选被同分精度回归拦住，表内性能不得当成可采用结果。**",
        "WS列池及原半叶修正版使用含新B1优化的多task快照，旧行保留原快照，不跨轮拼收益。", "",
        "“半叶分数留UB”的SPMD对照是已通过的原半叶GM版，其余行的具体参考见JSON。", "",
        "Score归属的“未激活”行没有执行重排；“同址开关”参考是身份映射诊断源。两者均不直接代表采用收益。", "",
        "| 实验 | B | 多task | 对照SPMD | 候选SPMD | 相对多task提升 | 相对对照净省 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["complete"]:
            lines.append(
                f"| {row['label']} | {row['batch']} | {row['multitask_us']:.2f} | "
                f"{row['control_spmd_us']:.2f} | {row['proposal_spmd_us']:.2f} | "
                f"{row['improvement_vs_multitask_pct']:+.2f}% | {row['saved_vs_control_us']:+.2f} |"
            )
        else:
            lines.append(f"| {row['label']} | {row['batch']} | 等待结果 | — | — | — | — |")
    lines += ["", "不能由某个中位数达标推导所有场景和采用回归均通过；最终目标尚未完成。"]
    (HERE / "GOAL10_FOCUS_PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
