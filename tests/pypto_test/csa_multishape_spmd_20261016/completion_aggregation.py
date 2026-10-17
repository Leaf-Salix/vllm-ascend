# SPDX-License-Identifier: Apache-2.0
"""在不增加数学节点先后关系的条件下，汇聚任务完成通知。

消费者原本等待某个前驱 task 的所有核。若某个生产者核的下一计算节点本来
就位于这些生产者之后，可以在该核本阶段计算结束后汇聚，再发布一个完成信号。
消费者等待这个信号即可。没有满足条件的生产者时，保留原逐核等待。
"""

import ast
import collections
import copy

from build_schedule_manifest import ancestor_sets
from prepare_static_layout import align
from validate_static_protocol import validate


def aggregate(schedule):
    nodes = [row for rows in schedule["per_core_sequences"].values() for row in rows]
    ids = {(str(row["core"]), row["dispatch_seq"]): i for i, row in enumerate(nodes)}
    members = collections.defaultdict(list)
    for i, row in enumerate(nodes):
        members[row["task"]].append(i)
    parents = {task["task_id"]: task["predecessors"] for task in schedule["tasks"]}
    used_tasks = {task for values in parents.values() for task in values}
    original = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            original[i].add(ids[(str(row["core"]), row["dispatch_seq"] - 1)])
        for task in parents[row["task"]]:
            original[i].update(members[task])
    order, ancestors = ancestor_sets(original)
    rank = {node: index for index, node in enumerate(order)}
    leaders = {}
    for task, producers in members.items():
        if task not in used_tasks:
            continue
        mask = sum(1 << node for node in producers)
        candidates = []
        for node in producers:
            row = nodes[node]
            following = ids.get((str(row["core"]), row["dispatch_seq"] + 1))
            if following is None or ancestors[following] & mask == mask:
                candidates.append(node)
        if candidates:
            leaders[task] = max(candidates, key=lambda node: (nodes[node].get("reference_end", 0), rank[node]))

    # 每个原成员拆成数学节点 M 和发布节点 P。原核内顺序是 P(prev)->M；
    # 汇聚只在 M->P 之间增加等待，不能简单把它移动到数学体之前。
    graph = [set() for _ in range(2 * len(nodes))]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            graph[2 * i].add(2 * ids[(str(row["core"]), row["dispatch_seq"] - 1)] + 1)
        for task in parents[row["task"]]:
            producers = [leaders[task]] if task in leaders else members[task]
            graph[2 * i].update(2 * producer + 1 for producer in producers)
        graph[2 * i + 1].add(2 * i)
        if leaders.get(row["task"]) == i:
            graph[2 * i + 1].update(2 * producer + 1 for producer in members[row["task"]] if producer != i)
    expanded_order, expanded_ancestors = ancestor_sets(graph)
    # 逐计算节点投影回原 DAG：没有漏依赖，也没有添加数学执行顺序。
    for i in range(len(nodes)):
        projection = sum(1 << j for j in range(len(nodes)) if expanded_ancestors[2 * i] & (1 << (2 * j)))
        if projection != ancestors[i]:
            raise ValueError(f"完成汇聚改变数学依赖偏序：成员 {i}")
    positions = {node: index for index, node in enumerate(expanded_order)}
    reduced = [set() for _ in graph]
    for node, predecessors in enumerate(graph):
        covered = 0
        for parent in sorted(predecessors, key=positions.get, reverse=True):
            if not covered & (1 << parent):
                reduced[node].add(parent)
                covered |= expanded_ancestors[parent] | (1 << parent)
    published = collections.defaultdict(set)
    pre_count, post_count = 0, 0
    for core, sequence in schedule["per_core_sequences"].items():
        observed = collections.defaultdict(int)
        for row in sequence:
            i = ids[(core, row["dispatch_seq"])]
            for phase, field in enumerate(("wait_for_core_progress", "gather_core_progress")):
                frontier = collections.defaultdict(int)
                for before in reduced[2 * i + phase]:
                    producer = nodes[before // 2]
                    source = str(producer["core"])
                    if source != core:
                        if before % 2 != 1:
                            raise ValueError("跨核等待必须指向可发布的节点")
                        frontier[source] = max(frontier[source], producer["dispatch_seq"])
                required = {source: step for source, step in sorted(frontier.items(), key=lambda x: int(x[0]))
                            if step > observed[source]}
                row[field] = required
                if phase:
                    post_count += len(required)
                else:
                    pre_count += len(required)
                for source, step in required.items():
                    observed[source] = step
                    published[source].add(step)
    for row in nodes:
        row["publish_completion"] = row["dispatch_seq"] in published[str(row["core"])]

    # 再核对序号翻译后的图；不能只证明约简前的抽象图。
    translated = [set() for _ in graph]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            translated[2 * i].add(2 * ids[(str(row["core"]), row["dispatch_seq"] - 1)] + 1)
        translated[2 * i + 1].add(2 * i)
        for phase, field in enumerate(("wait_for_core_progress", "gather_core_progress")):
            translated[2 * i + phase].update(2 * ids[(core, step)] + 1 for core, step in row[field].items())
    _, after = ancestor_sets(translated)
    if after != expanded_ancestors:
        raise ValueError("进度序号翻译改变了汇聚图的依赖关系")
    result = {"eligible_tasks": len(leaders), "total_tasks": len(members),
              "pre_waits": pre_count, "post_waits": post_count,
              "remote_core_progress_requirements": pre_count + post_count,
              "completion_publications": sum(map(len, published.values())),
              "original_wait_requirements": schedule["wait_plan"]["remote_core_progress_requirements"],
              "math_partial_order_equal": True, "translated_partial_order_equal": True,
              "global_barriers_added": 0}
    schedule["completion_aggregation"] = result
    return result


def extend_layout(schedule, layout):
    """8 字记录末两项是后置汇聚的起点与数量；复用原 waits Tensor。"""
    records, waits = [], []
    for batches in layout["batches"].values():
        for batch in batches:
            batch["record_offset_words"] = len(records)
            by_core = {row["core"]: row for row in batch["members"]}
            for core in range(72):
                row = by_core.get(core)
                if row is None:
                    records.extend([-1, 0, 0, 0, 0, 0, 0, 0])
                    continue
                pre, post = [], []
                for field, result in (("wait_for_core_progress", pre), ("gather_core_progress", post)):
                    result.extend([len(waits) // 2, len(row[field])])
                    for source, step in sorted(row[field].items(), key=lambda x: int(x[0])):
                        waits.extend([int(source), step])
                records.extend([row["logical_block"], int(row["original_subslot"] == "AIV1"),
                                row["dispatch_seq"], *pre, int(row["publish_completion"]), *post])
    workspace = layout["workspace"]
    workspace.update(records=records, waits=waits)
    offset = 0
    for name, count in (("progress", 72 * 64), ("records", len(records)), ("waits", len(waits))):
        workspace["sections"][name].update(offset_bytes=offset, bytes=count * 8, shape=[count])
        offset = align(offset + count * 8)
    for value in workspace["temporaries"].values():
        value["offset_bytes"] = offset
        offset = align(offset + value["bytes"])
    workspace["bytes"] = offset
    layout["protocol"].update(record_width=8, completion_aggregation=True)

    # 原验证器检查视图边界与原六字段；此处只借用这些结果。汇聚协议本身
    # 用 M/P 双阶段图证明，并在下方模型中显式执行 post_waits。
    base = copy.deepcopy(layout)
    base["protocol"]["record_width"] = 6
    base["workspace"]["records"] = [word for start in range(0, len(records), 8) for word in records[start:start + 6]]
    for batches in base["batches"].values():
        for batch in batches:
            batch["record_offset_words"] = batch["record_offset_words"] // 8 * 6
    basic = validate(schedule, base)
    basic.pop("progress_generation_model")
    basic["post_waits_explicitly_validated"] = phase_model(schedule)
    basic["math_partial_order_equal"] = schedule["completion_aggregation"]["math_partial_order_equal"]
    return basic


def phase_model(schedule):
    """检查跨代际双阶段协议；只有 post 等待完成才可发布本核进度。"""
    sequences = schedule["per_core_sequences"]
    flags = {core: 0 for core in sequences}
    reports = []
    for generation in range(1, 4):
        cursor = {core: -1 for core in sequences}
        completed = 0
        while any(cursor[core] < len(rows) * 2 for core, rows in sequences.items()):
            advanced = False
            # 固定逆核序扫描；它不假定真实设备按扫描顺序执行。
            for core in reversed(list(sequences)):
                position = cursor[core]
                rows = sequences[core]
                if position == -1:
                    if flags[core] // 1024 + 1 != generation:
                        raise ValueError("汇聚协议代际不一致")
                    flags[core] = generation * 1024
                elif position == len(rows) * 2:
                    continue
                else:
                    row = rows[position // 2]
                    phase = position % 2
                    field = "gather_core_progress" if phase else "wait_for_core_progress"
                    if any(flags[source] < generation * 1024 + step for source, step in row[field].items()):
                        continue
                    if phase and row["publish_completion"]:
                        flags[core] = generation * 1024 + row["dispatch_seq"]
                cursor[core] += 1
                completed += 1
                advanced = True
            if not advanced:
                raise ValueError("汇聚协议模型死锁")
        reports.append({"generation": generation, "steps": completed, "passed": True})
    return reports


def add_post_waits(source, layout):
    """只在原数学结束、发布前插入汇聚，保持读/写侧 payload 维护。"""
    module = ast.parse(source)
    for role in ("aic", "aiv"):
        fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "csa_static_" + role)
        for node in fn.body:
            if not isinstance(node, ast.If) or not ast.unparse(node.test).startswith("mapped_block_"):
                continue
            index = int(node.test.left.id.removeprefix("mapped_block_"))
            # 没有该阶段汇聚的核全部读到 count=0，不引入全核集合。
            if not any(row["gather_core_progress"] for row in layout["batches"][role][index]["members"]):
                continue
            code = f"""
gather_base_{index} = pl.cast(pl.tensor.read(records, [record_{index} + 6]), pl.INDEX)
gather_count_{index} = pl.cast(pl.tensor.read(records, [record_{index} + 7]), pl.INDEX)
for gather_{index} in pl.range(gather_count_{index}):
    gather_index_{index} = (gather_base_{index} + gather_{index}) * 2
    gather_core_{index} = pl.cast(pl.tensor.read(waits, [gather_index_{index}]), pl.INDEX)
    gather_step_{index} = pl.tensor.read(waits, [gather_index_{index} + 1])
    progress = csa_signal_wait(progress, gather_core_{index} * 128, pl.cast(generation_base + gather_step_{index}, pl.INT32))
"""
            node.body[-1:-1] = ast.parse(code).body
    ast.fix_missing_locations(module)
    return "# 完成通知汇聚候选；已核对原数学偏序，尚须真机与性能验证。\n" + ast.unparse(module) + "\n"
