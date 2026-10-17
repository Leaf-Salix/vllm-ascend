# SPDX-License-Identifier: Apache-2.0
"""把每档已经确定的分核/等待表编入控制流，保留业务 device metadata。

只改变调度描述的表达方式；数学体、发布序号、每条核间依赖和缓存维护保持。
将相同等待条件按生产者连续段归并，避免展开每个核的整份数学代码。
"""

import ast
import collections
import copy


def arithmetic_runs(values):
    """对整数键值做精确等差分段；返回的每个点都重新与原映射核对。"""
    points = sorted(values)
    runs, pos = [], 0
    while pos < len(points):
        start = points[pos]
        if pos + 1 == len(points):
            runs.append((start, start, 1, values[start], 0))
            break
        step = points[pos + 1] - start
        delta = values[points[pos + 1]] - values[start]
        end = pos + 1
        while (end + 1 < len(points) and points[end + 1] - points[end] == step
               and values[points[end + 1]] - values[points[end]] == delta):
            end += 1
        runs.append((start, points[end], step, values[start], delta))
        pos = end + 1
    restored = {}
    for begin, end, step, value, delta in runs:
        for key in range(begin, end + 1, step):
            if key in restored:
                raise ValueError("等差段出现重复键")
            restored[key] = value + (key - begin) // step * delta
    if restored != values:
        raise ValueError("等差段与原调度表不一致")
    return runs


def condition(cores):
    """核集合转为短区间条件；所有核心编号来自初始化时确定的分档计划。"""
    runs = arithmetic_runs({core: core for core in cores})
    parts = []
    for begin, end, step, _, _ in runs:
        if begin == end:
            parts.append(f"core_id == {begin}")
        else:
            inside = f"core_id >= {begin} and core_id <= {end}"
            if step != 1:
                inside += f" and (core_id - {begin}) % {step} == 0"
            parts.append(f"({inside})")
    return " or ".join(parts) if parts else "False"


def field_code(name, values, default):
    lines = [f"{name} = pl.cast({default}, pl.INDEX)"]
    for begin, end, step, value, delta in arithmetic_runs(values):
        expr = str(value) if delta == 0 else f"{value} + (core_id - {begin}) // {step} * {delta}"
        lines.extend([f"if {condition(range(begin, end + 1, step))}:",
                      f"    {name} = pl.cast({expr}, pl.INDEX)"])
    return lines


def wait_code(batch, number):
    # 转置消费者→前驱映射；相同消费者集合与 required 可共用一个生产者循环。
    pairs = collections.defaultdict(set)
    for row in batch["members"]:
        for other, required in row["wait_for_core_progress"].items():
            pairs[int(other), required].add(row["core"])
    groups = collections.defaultdict(dict)
    for (other, required), consumers in pairs.items():
        groups[tuple(sorted(consumers)), required][other] = other
    reconstructed = collections.defaultdict(dict)
    lines, loop_count = [], 0
    all_consumers = {row["core"] for row in batch["members"]}
    for (consumers, required), producers in sorted(groups.items()):
        gated = set(consumers) != all_consumers
        if gated:
            lines.append(f"if {condition(consumers)}:")
        indent = "    " if gated else ""
        for begin, end, step, _, _ in arithmetic_runs(producers):
            var = f"constant_predecessor_{number}_{loop_count}"
            if begin == end:
                lines.append(indent + f"progress = csa_signal_wait(progress, {begin * 128}, "
                             f"pl.cast(generation_base + {required}, pl.INT32))")
            else:
                lines.extend([indent + f"for {var} in pl.range({begin}, {end + 1}, {step}):",
                              indent + f"    progress = csa_signal_wait(progress, {var} * 128, "
                              f"pl.cast(generation_base + {required}, pl.INT32))"])
            loop_count += 1
            for consumer in consumers:
                for producer in range(begin, end + 1, step):
                    if producer in reconstructed[consumer]:
                        raise ValueError("压缩等待重复同一前驱")
                    reconstructed[consumer][producer] = required
    for row in batch["members"]:
        original = {int(k): v for k, v in row["wait_for_core_progress"].items()}
        if reconstructed[row["core"]] != original:
            raise ValueError("编译等待改变了核间依赖")
    return lines, {"wait_edges": sum(len(x) for x in reconstructed.values()),
                   "consumer_groups": len(groups), "producer_ranges": loop_count}


def compile_controls(source, layout):
    if layout["protocol"]["record_width"] != 6:
        raise ValueError("当前编译控制流不与通知汇聚实验组合")
    module = ast.parse(source)
    audit = []
    for role, batches in layout["batches"].items():
        fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "csa_static_" + role)
        rewritten, index = [], 0
        while index < len(fn.body):
            node = fn.body[index]
            if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.startswith("record_")):
                rewritten.append(node)
                index += 1
                continue
            number = int(node.targets[0].id.removeprefix("record_"))
            phase = fn.body[index + 3]
            if not isinstance(phase, ast.If) or phase.test.left.id != f"mapped_block_{number}":
                raise ValueError("生成器阶段边界改变")
            batch = batches[number]
            members = batch["members"]
            field_values = {
                "mapped_block": {r["core"]: r["logical_block"] for r in members},
                "subslot": {r["core"]: {"AIC": 0, "AIV0": 0, "AIV1": 1}[r["original_subslot"]]
                            for r in members},
                "done": {r["core"]: r["dispatch_seq"] for r in members},
                "publish": {r["core"]: int(r["publish_completion"]) for r in members},
            }
            header = []
            for field, values in field_values.items():
                header.extend(field_code(f"{field}_{number}", values, -1 if field == "mapped_block" else 0))
            header.append(f"block_{number} = pl.max(mapped_block_{number}, 0)")
            rewritten.extend(ast.parse("\n".join(header)).body)
            # 原前 5 句读取该核的记录，第 6 句遍历等待表；后面的数学和维护不改。
            if not isinstance(phase.body[5], ast.For):
                raise ValueError("等待表循环位置改变")
            preserved = copy.deepcopy(phase.body[6:])
            waits, detail = wait_code(batch, number)
            phase.body = ast.parse("\n".join(waits)).body + preserved
            rewritten.append(phase)
            audit.append({"role": role, "stage": number, "kernel": batch["kernel"],
                          "members": len(members), **detail, "math_and_fences_unchanged": True})
            index += 4
        fn.body = rewritten
    ast.fix_missing_locations(module)
    # 两个 Tensor 入参暂留以保持既有 ABI；核内不得再读取这两份描述表。
    for node in ast.walk(module):
        if (isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tensor.read"
                and node.args and isinstance(node.args[0], ast.Name)
                and node.args[0].id in {"records", "waits"}):
            raise ValueError("残留调度表读取")
    return ast.unparse(module) + "\n", audit
