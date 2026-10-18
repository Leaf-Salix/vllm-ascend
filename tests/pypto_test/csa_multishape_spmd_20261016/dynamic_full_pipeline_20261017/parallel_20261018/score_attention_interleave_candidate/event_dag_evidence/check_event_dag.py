#!/usr/bin/env python3
"""有界协议事件DAG：只建同一AIC/双AIV的源码同步偏序，不模拟硬件耗时或DMA完成。"""
from collections import defaultdict, deque
from itertools import product
from pathlib import Path
import json

def check_case(leaves, mask, early):
    score = [(leaf, step, count) for leaf, count in enumerate(leaves) for step in range(count)]
    count = len(mask)
    adj = defaultdict(list)
    degree = {}
    previous = {}
    emitted = set()
    edges = 0
    def edge(src, dst):
        nonlocal edges
        degree.setdefault(src, 0)
        degree.setdefault(dst, 0)
        adj[src].append(dst)
        degree[dst] += 1
        edges += 1
    def emit(owner, name, prerequisites=()):
        node = (owner, name)
        emitted.add(node)
        degree.setdefault(node, 0)
        if owner in previous:
            edge(previous[owner], node)
        for prerequisite in prerequisites:
            edge(prerequisite, node)
        previous[owner] = node
        return node
    def active(block):
        return 0 <= block < count and mask[block]

    # AIC每轮先完整Score动作（含叶尾ACK drain），后PV(r-2)再QK(r)。
    for r in range(max(len(score), count + 2)):
        emit("C", ("round", r))
        if r < len(score):
            leaf, step, leaf_steps = score[r]
            if step >= 2:
                emit("C", ("score_reuse_wait", r),
                     [(lane, ("score_ack", leaf, step - 2)) for lane in (0, 1)])
            emit("C", ("score_ready", leaf, step))
            if step + 1 == leaf_steps:
                for pending in range(max(0, leaf_steps - 2), leaf_steps):
                    emit("C", ("leaf_drain", leaf, pending),
                         [(lane, ("score_ack", leaf, pending)) for lane in (0, 1)])
        if r < count + 2:
            if active(r - 2):
                emit("C", ("pv_wait", r - 2),
                     [(lane, ("prob_ready", r - 2)) for lane in (0, 1)])
                emit("C", ("pv_ready", r - 2))
            if active(r):
                emit("C", ("kv_wait", r),
                     [(lane, ("kv_ready", r)) for lane in (0, 1)])
                emit("C", ("attention_score_ready", r))

    # AIV两lane都执行相同事件谓词；尾lane无query也仍读FIFO/ACK。
    for lane in (0, 1):
        for r in range(max(len(score), count + 3)):
            emit(lane, ("round", r))
            if r < count + 3:
                if active(r):
                    emit(lane, ("gather_store", r))
                if active(r - 1) and r <= count:
                    emit(lane, ("attention_score_wait", r - 1),
                         [("C", ("attention_score_ready", r - 1))])
                if early and active(r):
                    emit(lane, ("kv_ready", r))
                if active(r - 1) and r <= count:
                    emit(lane, ("prob_ready", r - 1))
                if not early and active(r):
                    emit(lane, ("kv_ready", r))
                if active(r - 3):
                    emit(lane, ("pv_wait", r - 3), [("C", ("pv_ready", r - 3))])
                # 末块即使mask无效仍有最终输出/重置，不引入新的跨核等待。
                if 0 <= r - 3 < count and (r - 3) % 5 == 4:
                    emit(lane, ("output_reset", (r - 3) // 5))
            if r < len(score):
                leaf, step, leaf_steps = score[r]
                emit(lane, ("score_wait", leaf, step), [("C", ("score_ready", leaf, step))])
                emit(lane, ("score_ack", leaf, step))
                emit(lane, ("score_math_after_ack", leaf, step))
                if step + 1 == leaf_steps:
                    emit(lane, ("leaf_root_store", leaf))

    # 所有wait源必须确实由三个角色发出，不能把缺失发布当成自由起点。
    assert set(degree) == emitted, (set(degree) - emitted)
    expected_event_nodes = {
        ("C", ("score_ready", leaf, step))
        for leaf, step, _ in score
    }
    expected_event_nodes |= {
        (lane, ("score_ack", leaf, step))
        for lane in (0, 1) for leaf, step, _ in score
    }
    assert expected_event_nodes <= degree.keys()
    ready = deque(node for node, incoming in degree.items() if incoming == 0)
    visited = 0
    indegree = degree.copy()
    while ready:
        node = ready.popleft()
        visited += 1
        for child in adj[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    assert visited == len(degree), (leaves, mask, early, [n for n, d in indegree.items() if d > 0][:8])
    return len(degree), edges

# 单叶每个1..8步；复叶只取FIFO两槽/叶重置关键边界，不扫工作负载shape。
patterns = [()] + [(n,) for n in range(1, 9)] + [
    (1, 1), (1, 8), (8, 1), (2, 2), (2, 3), (3, 2), (8, 8),
]
cases = 0
max_nodes = 0
max_edges = 0
for leaves in patterns:
    for blocks in (0, 5, 10):
        for mask in product((False, True), repeat=blocks):
            for early in (False, True):
                nodes, edges = check_case(leaves, mask, early)
                cases += 1
                max_nodes, max_edges = max(max_nodes, nodes), max(max_edges, edges)
result = {
    "passed": True,
    "cases": cases,
    "leaf_patterns": patterns,
    "attention_blocks": [0, 5, 10],
    "all_masks_per_block_count": True,
    "early_and_late_kv": True,
    "max_nodes": max_nodes,
    "max_edges": max_edges,
    "scope": "固定同组AIC+两个AIV的FFTS源码偏序；ROOT/TopK/PLAN外层另审，不包含编译器局部事件和DMA耗时",
    "conclusion": "枚举范围内没有新等待环；叶尾ACK只依赖当前AIV先完成的Attention，其等待来自前AIC tick。",
    "limits": [
        "未证明全系统任意wave metadata或异步流水的物理资源/局部事件正确性。",
        "未模拟调度性能，未把Scalar程序顺序等同MTE/FIX完成。",
        "原双槽Score和三槽GM协议的发布条件不变；AIC双Mat槽覆盖依赖须联合lowering核对。",
    ],
}
out = Path(__file__).resolve().parent / "event_dag_result.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("passed", "cases", "max_nodes", "max_edges")}, ensure_ascii=False))
