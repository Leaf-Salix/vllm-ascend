# SPDX-License-Identifier: Apache-2.0
"""把数学依赖及核内顺序变成单写者进度等待，保持完整偏序。"""

import collections


def ancestor_sets(predecessors):
    """返回拓扑序及每个节点的祖先位集；未知前驱和环均拒绝。"""
    children = collections.defaultdict(list)
    degree = [len(parents) for parents in predecessors]
    for node, parents in enumerate(predecessors):
        for parent in parents:
            if not isinstance(parent, int) or not 0 <= parent < len(predecessors):
                raise ValueError("成员依赖引用未知节点")
            children[parent].append(node)
    ready = collections.deque(i for i, count in enumerate(degree) if count == 0)
    order, ancestors = [], [0] * len(predecessors)
    while ready:
        node = ready.popleft()
        order.append(node)
        for parent in predecessors[node]:
            ancestors[node] |= ancestors[parent] | (1 << parent)
        for child in children[node]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(order) != len(predecessors):
        raise ValueError("成员依赖存在环")
    return order, ancestors


def rebuild_waits(per_core, tasks):
    """生成 notify/wait 的核序号，不引入阶段级全核屏障。

    先保留每个数学前驱的全部成员，再合并核内顺序并做传递约简。消费者
    只等待各远端核必要的最大序号；最终重新计算位集，确认未删任何偏序。
    这里只证明执行顺序，GM 信号与 payload 的缓存可见性由 signals 负责。
    """
    nodes = [member for sequence in per_core.values() for member in sequence]
    ids = {}
    members = collections.defaultdict(list)
    for core, sequence in per_core.items():
        for expected, row in enumerate(sequence, 1):
            if str(row["core"]) != core or row["dispatch_seq"] != expected:
                raise ValueError("核序列编号不连续或归属不符")
    for index, row in enumerate(nodes):
        key = str(row["core"]), row["dispatch_seq"]
        if key in ids:
            raise ValueError("核序列含重复成员")
        ids[key] = index
        members[row["task"]].append(index)
    parents = {task["task_id"]: task["predecessors"] for task in tasks}
    if len(parents) != len(tasks) or set(parents) != set(members):
        raise ValueError("任务描述与物理成员不完整对应")
    if any(parent not in parents for values in parents.values() for parent in values):
        raise ValueError("数学依赖引用未知任务")
    original = [set() for _ in nodes]
    for index, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            original[index].add(ids[str(row["core"]), row["dispatch_seq"] - 1])
        for parent in parents[row["task"]]:
            original[index].update(members[parent])
    order, ancestors = ancestor_sets(original)
    positions = {node: index for index, node in enumerate(order)}
    reduced = [set() for _ in nodes]
    for node, predecessors in enumerate(original):
        covered = 0
        for parent in sorted(predecessors, key=positions.get, reverse=True):
            if not covered & (1 << parent):
                reduced[node].add(parent)
                covered |= ancestors[parent] | (1 << parent)

    published = collections.defaultdict(set)
    for core, sequence in per_core.items():
        observed = collections.defaultdict(int)
        for row in sequence:
            required = collections.defaultdict(int)
            for parent in reduced[ids[core, row["dispatch_seq"]]]:
                before = nodes[parent]
                source = str(before["core"])
                if source != core:
                    required[source] = max(required[source], before["dispatch_seq"])
            row["wait_for_core_progress"] = {
                source: step
                for source, step in sorted(required.items(), key=lambda pair: int(pair[0]))
                if step > observed[source]
            }
            for source, step in row["wait_for_core_progress"].items():
                observed[source] = step
                published[source].add(step)
    planned = [set() for _ in nodes]
    for index, row in enumerate(nodes):
        core = str(row["core"])
        row["publish_completion"] = row["dispatch_seq"] in published[core]
        if row["dispatch_seq"] > 1:
            planned[index].add(ids[core, row["dispatch_seq"] - 1])
        planned[index].update(ids[source, step] for source, step in row["wait_for_core_progress"].items())
    _, after = ancestor_sets(planned)
    if after != ancestors:
        raise ValueError("生成等待表后依赖偏序改变")
    return {
        "remote_core_progress_requirements": sum(len(row["wait_for_core_progress"]) for row in nodes),
        "completion_publications": sum(row["publish_completion"] for row in nodes),
        "original_member_edges": sum(map(len, original)),
        "planned_member_edges": sum(map(len, planned)),
        "identical_dependency_partial_order": True,
        "global_barriers_added": 0,
    }
