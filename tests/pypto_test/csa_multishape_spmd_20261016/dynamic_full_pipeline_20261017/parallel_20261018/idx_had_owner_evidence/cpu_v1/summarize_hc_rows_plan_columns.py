# SPDX-License-Identifier: Apache-2.0
"""读取已有七档结果，核对完整L1、源码和实际策略，不提交新测试。

两档选同进程矩阵中的HC行依赖开启分支，其他五档是显式开启的成对测量。
唯一允许的后续源码差别是已测HC策略从显式选择变成默认选择；其余AST均须
一致。记录原/当前默认值，不用忽略任意代码差异的方式混入旧版本。
"""

import ast
import json
import statistics
from pathlib import Path

from summarize_output_groups import raw_samples

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def source_contract(path):
    tree = ast.parse(path.read_text())
    default = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    node.body.pop(0)
        if path.name == "policy.py" and isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "hc_row_dependencies":
                assert isinstance(node.value, ast.Constant) and type(node.value.value) is bool
                default = node.value.value
                node.value = ast.Constant(None)
    return ast.dump(tree, include_attributes=False), default


def main():
    jobs = json.loads((HERE / "hc_rows_plan_columns_jobs.json").read_text())["jobs"]
    old = json.loads((HERE / "output_groups_seven.json").read_text())["rows"]
    rows = []
    for job in jobs:
        directory = HERE / job["result"]
        report = json.loads((directory / "result.json").read_text())
        assert report["passed"] and report["stage"] == "complete"
        assert report["graph_replay_validated"] and report["post_l1_protocol"]
        defaults = {}
        for path in PKG.glob("*.py"):
            current, new_default = source_contract(path)
            saved, old_default = source_contract(directory / "source" / path.name)
            assert current == saved, (job["task"], path.name)
            if path.name == "policy.py":
                defaults = dict(measured_source=old_default, current_source=new_default)
        side = job.get("variant", "single_spmd")
        timing = report[job["matrix"]] if "matrix" in job else report["l1"]
        baseline_samples = raw_samples(timing["windows"], "dynamic_pto")
        spmd_samples = raw_samples(timing["windows"], side)
        baseline, spmd = statistics.median(baseline_samples), statistics.median(spmd_samples)
        assert abs(baseline - timing["statistics_us"]["dynamic_pto"]["median_us"]) < 1e-6
        assert abs(spmd - timing["statistics_us"][side]["median_us"]) < 1e-6
        policy = timing["policies"][side] if "matrix" in job else report["policy"]
        assert policy["HC_ROW_DEPENDENCIES"] and policy["OUTPUT_GROUP_PIPELINE"]
        previous = next(x for x in old if (x["history"], x["batch"]) == (job["history"], job["batch"]))
        rows.append(dict(
            **job, query_tokens=6, passed=True, dynamic_pto_us=baseline, single_spmd_us=spmd,
            latency_reduction_pct=(1 - spmd / baseline) * 100, target_met=spmd <= baseline * 0.7,
            samples_baseline_us=baseline_samples, samples_spmd_us=spmd_samples, policy=policy,
            source_ast_equal_except_documentation_and_hc_default=True, hc_defaults=defaults,
            previous_output_groups_us=previous["single_spmd_us"],
            historical_static_us=previous["historical_static_us"],
        ))
    result = dict(
        scope="源级动态HC行依赖＋页有效列；固定S6；同轮完整L1五窗中位数",
        passed=True, performance_target_met=all(x["target_met"] for x in rows), rows=rows,
        wait_backoff="100tick测试编译hook，正式binding尚未默认安装",
        not_validated=["mixed-query完整CSA", "padding完整CSA", "请求生命周期", "prefix共享", "连续状态演进"],
        metadata_placement="不强制全部位于SPMD；移出时新增成本计入完整链路",
    )
    (HERE / "hc_rows_plan_columns_seven.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# HC行依赖与页检查有效列：动态CSA七档",
        "",
        "单位μs，S=6；每行同设备同进程的原分task PTO对照，完整L1首尾五窗中位数。",
        "测试包含内部等待、metadata与阶段空隙，排除外围首尾调度。使用正式W8A8权重。",
        "",
        "| 历史 / B | 同轮分task PTO | 动态单SPMD | 相对分task | 历史静态SPMD（跨轮） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        old_static = "未测" if row["historical_static_us"] is None else f'{row["historical_static_us"]:.2f}'
        diff = row["latency_reduction_pct"]
        relation = f'{"快" if diff > 0 else "慢"} {abs(diff):.2f}%'
        lines.append(
            f'| {row["history"] // 1024}K / B{row["batch"]} | {row["dynamic_pto_us"]:.2f} | '
            f'{row["single_spmd_us"]:.2f} | {relation} | {old_static} |'
        )
    lines += [
        "",
        "七档固定S6的输出、TopK/score、cache/state、图重放、保护区和代际均通过。",
        "功能通过不等于30%性能目标完成，也不替代mixed-query、padding及连续状态验证。",
        "",
        "历史静态为另一轮三窗结果，只作参考，不能把全部差距归因于动态轴；原静态表无128K/B28。",
        "当前数据仍启用100tick失败轮询退避测试hook，正式绑定尚未默认启用。",
        "",
        "同进程HC消融：128K/B16为934.14→914.22（降低2.13%），8K/B16为738.72→716.32（降低3.03%）。",
        "其它五档没有同轮HC开关消融，不把跨轮差值写成同轮收益。",
        "",
        "原始任务、逐窗样本、策略及源码默认值记录：[JSON](hc_rows_plan_columns_seven.json)。",
        "",
    ]
    for row in rows:
        lines.append(f'- {row["history"] // 1024}K/B{row["batch"]}：[{row["task"]}]({row["result"]}/result.json)。')
    (HERE / "HC_ROWS_PLAN_COLUMNS_SEVEN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    for row in rows:
        print(row["history"], row["batch"], row["dynamic_pto_us"], row["single_spmd_us"], row["latency_reduction_pct"])


if __name__ == "__main__":
    main()
