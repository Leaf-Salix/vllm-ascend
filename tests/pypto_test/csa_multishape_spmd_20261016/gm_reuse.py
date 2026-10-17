# SPDX-License-Identifier: Apache-2.0
"""按完整任务 DAG 复用等大小临时 GM；不增加等待或改数学核体。

必须是旧 allocation 的每一个使用者，都在新 allocation 的每一个使用者之前。
不拿 trace 时间、task ordinal 或单核顺序代替这个证明。信号表、原生入参、
FFTS 工作区和没有调用引用的 allocation 保持独占。

只合并相同字节范围：PyPTO Torch 入口允许完整别名，拒绝部分可写重叠。
这些 Tensor 是本次调用内部 scratch，调用结束后不承诺各中间值仍然保留；
正式输出、Top-K 和六份状态不参与复用。
"""

import copy
import math

from bind_orchestration import DTYPE_BYTES
from build_schedule_manifest import ancestor_sets


def lifetimes(schedule, layout):
    tasks = schedule["tasks"]
    indices = {task["task_id"]: i for i, task in enumerate(tasks)}
    _, ancestors = ancestor_sets([
        {indices[parent] for parent in task["predecessors"]} for task in tasks])
    users = {name: set() for name in layout["workspace"]["temporaries"]}
    for task, call in schedule["call_bindings"].items():
        for argument in call["arguments"]:
            ref = argument["value"]
            if isinstance(ref, dict) and ref.get("allocation") in users:
                users[ref["allocation"]].add(indices[task])
                allocation = layout["workspace"]["temporaries"][ref["allocation"]]
                if (ref["offset"] != 0 or math.prod(ref["shape"]) * DTYPE_BYTES[ref["dtype"]]
                        != allocation["bytes"]):
                    raise ValueError("复用候选含部分临时视图，不能直接用完整别名规则")
    excluded = {name: "没有可证明的调用使用范围" for name, use in users.items() if not use}
    for event in layout["existing_internal_sync"]:
        for operation in event["operations"]:
            ref = operation.get("bound_workspace", {})
            name = ref.get("allocation")
            if name in users:
                users[name].add(indices[event["task"]])
                excluded[name] = "内核 FFTS 同步工作区独占"

    def precedes(left, right):
        return (left not in excluded and right not in excluded
                and bool(users[left]) and bool(users[right])
                and all(ancestors[b] & (1 << a) for a in users[left] for b in users[right]))

    return users, excluded, precedes


def validate_reuse(schedule, layout):
    """直接检查最终地址布局，拒绝未证明的重叠，而非信任分组器的输出。"""
    workspace = layout["workspace"]
    users, excluded, precedes = lifetimes(schedule, layout)
    regions = {**workspace["sections"], **workspace["temporaries"]}
    names = list(regions)
    overlaps = []
    for index, name in enumerate(names):
        ref = regions[name]
        start, end = ref["offset_bytes"], ref["offset_bytes"] + ref["bytes"]
        if start < 0 or start % workspace["alignment"] or end > workspace["bytes"]:
            raise ValueError(f"复用工作区边界/对齐错误：{name}")
        for other in names[:index]:
            prev = regions[other]
            pstart, pend = prev["offset_bytes"], prev["offset_bytes"] + prev["bytes"]
            if max(start, pstart) >= min(end, pend):
                continue
            if name not in users or other not in users or start != pstart or end != pend:
                raise ValueError(f"不允许控制区或部分范围重叠：{name}/{other}")
            if precedes(name, other):
                before, after = name, other
            elif precedes(other, name):
                before, after = other, name
            else:
                raise ValueError(f"任务 DAG 未证明生命周期分离：{name}/{other}")
            overlaps.append({"before": before, "after": after,
                             "all_user_pairs_proved": len(users[before]) * len(users[after])})
    return {"workspace_lifetime_safe_reuse": "PASS", "complete_span_aliases_only": True,
            "proved_pairs": overlaps, "excluded": excluded,
            "task_users": {name: [schedule["tasks"][i]["task_id"] for i in sorted(use)]
                           for name, use in users.items()},
            "added_dependencies": 0, "native_allocations_reused": False,
            "scratch_post_call_values_preserved": False,
            "scope": "CPU DAG、地址和完整别名证明；不替代设备数值、可见性或性能检查"}


def reuse_equal_spans(schedule, layout):
    """只改 allocation 起始地址；保留每份逻辑 Tensor 的 dtype/shape/stride。"""
    before = copy.deepcopy(layout["workspace"])
    workspace = layout["workspace"]
    _, excluded, precedes = lifetimes(schedule, layout)
    slots = []
    allocations = workspace["temporaries"]
    for name in sorted(allocations, key=lambda name: (-allocations[name]["bytes"], name)):
        slot = next((slot for slot in slots
                     if name not in excluded
                     and allocations[slot[0]]["bytes"] == allocations[name]["bytes"]
                     and all(precedes(name, other) or precedes(other, name) for other in slot)), None)
        if slot is None:
            slots.append([name])
        else:
            slot.append(name)
    alignment = workspace["alignment"]

    def align(size):
        return (size + alignment - 1) // alignment * alignment

    cursor = max(align(ref["offset_bytes"] + ref["bytes"])
                 for ref in workspace["sections"].values())
    for slot in slots:
        for name in slot:
            allocations[name]["offset_bytes"] = cursor
        cursor += align(allocations[slot[0]]["bytes"])
    workspace.update(bytes=cursor, reuse="equal_span_strict_task_dag")
    proof = validate_reuse(schedule, layout)
    proof.update(original_workspace_bytes=before["bytes"], reused_workspace_bytes=cursor,
                 saved_bytes=before["bytes"] - cursor, slots=slots)
    # 控制区位置和表内容必须原样保留，不能借内存优化悄悄修改跨核同步。
    for field in ("sections", "records", "waits"):
        if workspace[field] != before[field]:
            raise ValueError("复用改变了控制区")
    for name, previous in before["temporaries"].items():
        current = allocations[name]
        if {k: v for k, v in current.items() if k != "offset_bytes"} != {
                k: v for k, v in previous.items() if k != "offset_bytes"}:
            raise ValueError("复用改变了数学 Tensor 描述符")
    return proof
