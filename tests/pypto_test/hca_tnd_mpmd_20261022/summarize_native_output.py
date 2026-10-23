# SPDX-License-Identifier: Apache-2.0
"""汇总已有的Native数学修正与O_B分块结果，不启动设备或补测。

只用同一任务中的control/candidate计算收益。Native cache/state严格比较仍
单列保留；输出通过、服务接管通过均不能替代整模型token精度验收。
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def all_passed(value):
    if isinstance(value, list):
        return all(all_passed(v) for v in value)
    if "status" in value:
        return value["status"] == "PASS"
    return all(all_passed(v) for v in value.values())


def collect(manifest):
    path = HERE / manifest
    rows = []
    if not path.exists():
        return rows
    for task in json.loads(path.read_text()):
        source = HERE / "runs" / task["name"] / "result.json"
        if not source.exists():
            rows.append({"name": task["name"], "task_id": task["task_id"], "stage": "no_result_yet"})
            continue
        result = json.loads(source.read_text())
        row = {
            "name": task["name"], "task_id": task["task_id"],
            "source": str(source.relative_to(HERE)),
            "stage": result["stage"], "passed_all_strict_checks": result["passed"],
            "shared_toolchain": result.get("shared_toolchain"),
            "queries": result.get("query_lengths"), "histories": result.get("history_lengths"),
        }
        if "pto_native" in result:
            row["native_output"] = result["pto_native"]["output"]
            row["strict_native_storage_pass"] = all_passed(
                {k: v for k, v in result["pto_native"].items() if k != "output"})
            row["repeat_exact"] = all_passed(result["repeat_eager"])
            row["guards_pass"] = all(all_passed(result[name]) for name in
                                     ("native_guards", "pto_guards", "repeat_guards"))
        if "pto_reference" in result:
            row["control_output_exact"] = all_passed(result["pto_reference"]["output"])
            row["control_storage_exact"] = all_passed(
                {k: v for k, v in result["pto_reference"].items() if k != "output"})
        if "output_math_transition" in result:
            row["output_math_transition"] = result["output_math_transition"]
        if "paired_l1" in result:
            pair = result["paired_l1"]
            row["paired_l1"] = {side: {k: value[k] for k in ("samples_us", "median_us")}
                                for side, value in pair.items()}
            old, new = (pair[side]["median_us"] for side in ("control", "candidate"))
            row["latency_reduction_us"] = old - new
            row["latency_reduction_pct"] = (old - new) / old * 100
            row["speedup_pct_user_definition"] = (old / new - 1) * 100
        if "service_validation" in result:
            row["service_pass"] = result["service_validation"]["passed"]
            row["service_events"] = [v["events"] for v in result["service_validation"]["calls"]]
        if "traceback" in result:
            row["failure"] = result["traceback"]
        rows.append(row)
    return rows


def main():
    report = {
        "scope": "已有直接调用配对L1；无ACLGraph、无新测量；Native/PTO目标未验收",
        "original_math": collect("NATIVE_OUTPUT_PERF_TASKS.json")
                         + collect("NATIVE_OUTPUT_COVERAGE_TASKS.json"),
        "integer_split": collect("NATIVE_OUTPUT_SPLIT_TASKS.json"),
        "range_policy": collect("NATIVE_OUTPUT_RANGES_TASKS.json"),
    }
    (HERE / "NATIVE_OUTPUT_SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# HCA输出数学修正与O_B分块实测", "",
             "全部为同任务、同SDK的直接调用L1首尾，单位μs。比例按用户定义`control/candidate−1`。",
             "旧分组量化与Native整token量化的输出允许改变；三组cache/state必须逐bit保持。",
             "Native storage严格1e−5检查仍有差异，因此以下均不是完整精度或30%目标验收。", ""]
    for key, title in (("original_math", "旧分组尾链 → Native数学尾链"),
                       ("integer_split", "Native数学尾链 → N256、K分两段"),
                       ("range_policy", "区间策略补验：T>96使用N256、K分两段")):
        lines += ["## " + title, "", "| 场景 | control | candidate | 性能提升 | 输出容差 | 控制storage精确 | 窗口数 |",
                  "|---|---:|---:|---:|---|---|---:|"]
        for row in report[key]:
            if "paired_l1" not in row:
                continue
            q, h = row["queries"], row["histories"]
            label = f"{max(h) // 1024}K/B{len(q)}/" + (f"S{q[0]}" if len(set(q)) == 1 else f"混长T{sum(q)}")
            p = row["paired_l1"]
            lines.append(f"| {label} | {p['control']['median_us']:.2f} | {p['candidate']['median_us']:.2f} "
                         f"| {row['speedup_pct_user_definition']:+.2f}% | {row['native_output']['status']} "
                         f"| {row['control_storage_exact']} | {len(p['candidate']['samples_us'])} |")
        lines.append("")
    services = [r for r in report["original_math"] if "service_pass" in r]
    lines += ["## 服务与记录边界", ""]
    for row in services:
        lines.append(f"- `{row['name']}`：接管及三次调用顺序{'通过' if row['service_pass'] else '失败'}；"
                     f"Native输出{row['native_output']['status']}。")
    lines += ["", "任务ID、全部样本、失败记录和原始报告路径见同名JSON。",
              "`coverage_v1`中五项提交遗漏`--batch`，在fixture构造前被拒绝；只修命令后重提v2，未掩盖失败。",
              "服务验证覆盖真实custom op及KV钩子，不包含FFN、整网token或实际PD传输。", ""]
    (HERE / "NATIVE_OUTPUT_SUMMARY.md").write_text("\n".join(lines))
    print(HERE / "NATIVE_OUTPUT_SUMMARY.md")


if __name__ == "__main__":
    main()
