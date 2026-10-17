# SPDX-License-Identifier: Apache-2.0
"""汇总当前输出组流水七档，重算原始L1首尾并区分功能与性能门槛。

只读取已完成任务，不提交测试。矩阵只取正式策略选中的候选；不能把同一
进程中的其它策略、未完成结果或旧源码混入本版本。源码一致性用去除文档
字符串后的AST比较，不计算hash。固定S6通过不代表所有动态场景通过。
"""

import ast
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


class WithoutDocs(ast.NodeTransformer):
    def visit_Module(self, node):
        return self.strip(node)

    def visit_FunctionDef(self, node):
        return self.strip(node)

    def visit_ClassDef(self, node):
        return self.strip(node)

    def strip(self, node):
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = node.body[1:]
        return self.generic_visit(node)


def syntax(path):
    return ast.dump(WithoutDocs().visit(ast.parse(path.read_text())))


def raw_samples(windows, side):
    samples = []
    for window in windows:
        if window["side"] != side:
            continue
        raw = json.loads(Path(window["raw"]).read_text())
        records = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1 and len({row[0] for row in records}) == 72
        if side != "dynamic_pto":
            assert len(records) == 72 and len({row[1] for row in records}) == 1
        measured = (
            (max(row[4] for row in records) - min(row[3] for row in records)) * 1e6 / raw["metadata"]["clock_freq_hz"]
        )
        assert abs(measured - window["incore_span_us"]) < 1e-6
        samples.append(measured)
    assert len(samples) == 5, (side, samples)
    return samples


def main():
    jobs = json.loads((HERE / "output_group_jobs.json").read_text())["jobs"]
    static_rows = json.loads((HERE.parent / "selected_shape_policies.json").read_text())["rows"]
    rows = []
    for job in jobs:
        directory = HERE / job["result"]
        report = json.loads((directory / "result.json").read_text())
        assert report["passed"] and report["stage"] == "complete", (job["task"], report["stage"])
        assert report["graph_replay_validated"] and report["post_l1_protocol"]
        for path in PKG.glob("*.py"):
            assert syntax(path) == syntax(directory / "source" / path.name), (job["task"], path.name)
        side = job.get("variant", "single_spmd")
        timing = report["output_policy_matrix"] if "variant" in job else report["l1"]
        samples_b = raw_samples(timing["windows"], "dynamic_pto")
        samples_s = raw_samples(timing["windows"], side)
        baseline, candidate = statistics.median(samples_b), statistics.median(samples_s)
        assert abs(baseline - timing["statistics_us"]["dynamic_pto"]["median_us"]) < 1e-6
        assert abs(candidate - timing["statistics_us"][side]["median_us"]) < 1e-6
        old = next((r for r in static_rows if r["history"] == job["history"] and r["batch"] == job["batch"]), None)
        policy = timing["policies"][side] if "variant" in job else report["policy"]
        assert policy["OUTPUT_GROUP_PIPELINE"]
        rows.append(
            dict(
                **job,
                query_tokens=6,
                passed=True,
                graph_replay_validated=True,
                formal_source_ast_equal=True,
                dynamic_pto_us=baseline,
                single_spmd_us=candidate,
                latency_reduction_pct=(1 - candidate / baseline) * 100,
                target_met=candidate <= baseline * 0.7,
                samples_baseline_us=samples_b,
                samples_spmd_us=samples_s,
                policy=policy,
                historical_static_us=old["l1_us"] if old else None,
            )
        )
    result = dict(
        scope="同一正式动态源码、S6、正式权重、完整L1五窗中位数",
        passed=True,
        performance_target_met=all(r["target_met"] for r in rows),
        rows=rows,
        not_validated=["mixed-query完整链路", "padding完整链路", "请求生命周期", "prefix共享", "连续状态轨迹"],
        wait_backoff="测试编译hook启用100tick退避；正式绑定未默认安装hook",
    )
    (HERE / "output_groups_seven.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 输出按组就绪流水：正式动态 CSA 七档",
        "",
        "2026-10-17；均为S6，单位μs。每行同轮、同进程完整L1首尾五窗中位数，包含内部等待和metadata工作。",
        "",
        "| 历史 / B | 同轮分task PTO | 动态单SPMD | 相对分task | 历史静态SPMD（跨轮） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        old = "未测" if row["historical_static_us"] is None else f"{row['historical_static_us']:.2f}"
        change = ("快 " if row["latency_reduction_pct"] > 0 else "慢 ") + f"{abs(row['latency_reduction_pct']):.2f}%"
        lines.append(
            f"| {row['history'] // 1024}K / B{row['batch']} | {row['dynamic_pto_us']:.2f} | "
            f"{row['single_spmd_us']:.2f} | {change} | {old} |"
        )
    lines += [
        "",
        "七档固定S6的输出、Top-K、cache/state、图重放、信号与保护检查均通过。"
        "功能通过和30%性能门槛分开，具体达标情况见JSON，不以平均收益替代逐档门槛。",
        "",
        "B16/B32引用同进程策略矩阵的正式选中分支，其余五档分别与原分task PTO成对测量；"
        "各候选源文件去除文档后的AST一致。原始L1的第3/4下标首尾重新计算，与记录中位数一致。",
        "",
        "历史静态数据来自另一轮，不能把全部差距归因于dynamic shape。原静态表包含8K/B24，"
        "没有当前新增的128K/B28。未覆盖的动态场景与测试退避hook边界见JSON及主日志。",
        "",
        "原始任务与逐窗样本：[output_groups_seven.json](output_groups_seven.json)。",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['history'] // 1024}K/B{row['batch']}：[{row['task']}]({row['result']}/result.json)。")
    (HERE / "OUTPUT_GROUPS_SEVEN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    for row in rows:
        print(row["history"], row["batch"], row["dynamic_pto_us"], row["single_spmd_us"], row["latency_reduction_pct"])


if __name__ == "__main__":
    main()
