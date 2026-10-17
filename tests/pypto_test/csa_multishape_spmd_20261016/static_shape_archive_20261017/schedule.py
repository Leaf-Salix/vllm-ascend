# SPDX-License-Identifier: Apache-2.0
"""按源码任务图和显式模型成本分核，不读取历史时间线或测量文件。"""

import collections
import copy
import math

from .wait_plan import rebuild_waits

AIC_CORES = 24
AIV_CORES = 48
CORE_COUNT = AIC_CORES + AIV_CORES
SLOTS = ("AIC", "AIV0", "AIV1")
DEFAULT_MEMBER_COST = 10.0


def earliest_fit(calendars, cores, release, durations):
    """在相关核日历寻找共同起点；MIX 三成员始终占用同一物理组。"""
    start = release
    while True:
        ends = [
            end
            for core, duration in zip(cores, durations)
            for begin, end in calendars[core]
            if start < end and start + duration > begin
        ]
        if not ends:
            return start
        start = min(ends)


def _resolve_costs(schedule, member_costs):
    """优先级：本轮(ordinal,slot)、(kernel,slot)、kernel、默认常量。

    所有值都是正有限的调度权重，没有隐式 +1 或微秒单位。即使调用者从
    实测数据离线取得权重，也不能把分核模型的跨度宣称为设备实测耗时。
    """
    supplied = dict(member_costs or {})
    for value in supplied.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("成员成本必须是正有限数")
    valid, costs, audit = set(), {}, []
    for task in schedule["tasks"]:
        task_id = task["task_id"]
        ordinal = schedule["call_bindings"][task_id]["ordinal"]
        for slot, name in zip(SLOTS, task["kernel_names"]):
            if not name:
                continue
            candidates = ((ordinal, slot), (name, slot), name)
            valid.update(candidates)
            key = next((key for key in candidates if key in supplied), None)
            cost = float(supplied[key]) if key is not None else DEFAULT_MEMBER_COST
            costs[task_id, slot] = cost
            audit.append(
                {
                    "task": task_id,
                    "call_ordinal": ordinal,
                    "kernel": name,
                    "slot": slot,
                    "model_cost": cost,
                    "source": "explicit_model_weight" if key is not None else "default_unmeasured_model_weight",
                    "source_key": key,
                    "measured": False,
                }
            )
    if set(supplied) - valid:
        raise ValueError("成员成本引用了当前编排没有的核或角色")
    return costs, audit


def assign_members(schedule, templates, *, policy="earliest_ready", member_costs=None):
    """更新完整 schedule；templates[(task_id,slot)] 只含源码成员描述。"""
    if policy not in {"earliest_ready", "critical_path"}:
        raise ValueError("未知分核策略")
    tasks = {task["task_id"]: task for task in schedule["tasks"]}
    if not tasks or len(tasks) != len(schedule["tasks"]):
        raise ValueError("数学任务为空或编号重复")
    durations, cost_audit = _resolve_costs(schedule, member_costs)
    if set(templates) != set(durations):
        raise ValueError("源码成员描述未完整覆盖各任务角色")
    children = collections.defaultdict(list)
    for task_id, task in tasks.items():
        for parent in task["predecessors"]:
            if parent not in tasks:
                raise ValueError("数学任务引用未知前驱")
            children[parent].append(task_id)
    ranks, active = {}, set()

    def rank(task_id):
        if task_id in active:
            raise ValueError("数学任务图含环")
        if task_id not in ranks:
            active.add(task_id)
            task = tasks[task_id]
            slots = [slot for slot, name in zip(SLOTS, task["kernel_names"]) if name]
            capacity = AIC_CORES if task["kernel_names"][0] else AIV_CORES
            own = max(durations[task_id, slot] for slot in slots) * math.ceil(task["block_num"] / capacity)
            ranks[task_id] = own + max((rank(child) for child in children[task_id]), default=0)
            active.remove(task_id)
        return ranks[task_id]

    for task_id in tasks:
        rank(task_id)
    calendars = {core: [] for core in range(CORE_COUNT)}
    queues = {str(core): [] for core in range(CORE_COUNT)}
    finished = {}

    def role_cores(task):
        if task["kernel_names"][0] and any(task["kernel_names"][1:]):
            if not all(task["kernel_names"]):
                raise ValueError("MIX 必须完整包含 AIC、AIV0、AIV1")
            return [(group, AIC_CORES + 2 * group, AIC_CORES + 2 * group + 1) for group in range(AIC_CORES)]
        if task["kernel_names"][0]:
            return [(core,) for core in range(AIC_CORES)]
        if sum(bool(name) for name in task["kernel_names"]) != 1:
            raise ValueError("不支持没有 AIC 的多成员任务")
        return [(core,) for core in range(AIC_CORES, CORE_COUNT)]

    while len(finished) < len(tasks):
        ready = [
            task_id
            for task_id, task in tasks.items()
            if task_id not in finished and all(parent in finished for parent in task["predecessors"])
        ]
        if not ready:
            raise ValueError("数学任务图不能推进")

        def priority(task_id):
            task = tasks[task_id]
            release = max((finished[parent] for parent in task["predecessors"]), default=0)
            elapsed = [durations[task_id, slot] for slot, name in zip(SLOTS, task["kernel_names"]) if name]
            start = min(earliest_fit(calendars, cores, release, elapsed) for cores in role_cores(task))
            return (
                start if policy == "earliest_ready" else 0,
                -ranks[task_id],
                schedule["call_bindings"][task_id]["ordinal"],
            )

        task_id = min(ready, key=priority)
        task = tasks[task_id]
        slots = [slot for slot, name in zip(SLOTS, task["kernel_names"]) if name]
        elapsed = [durations[task_id, slot] for slot in slots]
        release = max((finished[parent] for parent in task["predecessors"]), default=0)
        completion = release
        for block in range(task["block_num"]):
            start, cores = min((earliest_fit(calendars, cores, release, elapsed), cores) for cores in role_cores(task))
            for slot, core, duration in zip(slots, cores, elapsed):
                end = start + duration
                row = copy.deepcopy(templates[task_id, slot])
                row.update(
                    core=core,
                    logical_block=block,
                    logical_block_num=task["block_num"],
                    logical_block_stride=task["block_num"],
                    start=start,
                    end=end,
                    model_start=start,
                    model_end=end,
                    time_fields_are_unmeasured_schedule_model=True,
                )
                queues[str(core)].append(row)
                calendars[core].append((start, end))
                calendars[core].sort()
                completion = max(completion, end)
        finished[task_id] = completion
    # 不携带空核队列，layout 按真实参与成员构建批次；硬件核总数仍固定为72。
    queues = {core: sequence for core, sequence in queues.items() if sequence}
    for sequence in queues.values():
        sequence.sort(key=lambda row: (row["model_start"], row["model_end"]))
        for index, row in enumerate(sequence, 1):
            row["dispatch_seq"] = index
    coverage = collections.Counter(
        (row["task"], row["original_subslot"], row["logical_block"]) for sequence in queues.values() for row in sequence
    )
    expected = collections.Counter(
        (task_id, slot, block)
        for task_id, task in tasks.items()
        for slot, name in zip(SLOTS, task["kernel_names"])
        if name
        for block in range(task["block_num"])
    )
    if coverage != expected:
        raise ValueError("逻辑分片遗漏或重复")
    groups = collections.defaultdict(dict)
    for sequence in queues.values():
        for row in sequence:
            groups[row["task"], row["logical_block"]][row["original_subslot"]] = row["core"]
    for (task_id, _), members in groups.items():
        if all(tasks[task_id]["kernel_names"]):
            group = members["AIC"]
            if members["AIV0"] != AIC_CORES + 2 * group or members["AIV1"] != AIC_CORES + 2 * group + 1:
                raise ValueError("MIX 分片未分配到同一物理核组")
    schedule["per_core_sequences"] = queues
    schedule["wait_plan"] = rebuild_waits(queues, list(tasks.values()))
    schedule["summary"].update(physical_members=sum(map(len, queues.values())), participating_cores=len(queues))
    schedule["balanced_schedule"] = {
        "policy": policy,
        "estimator": "显式模型权重；不代表实测微秒或性能承诺",
        "model_span": max(finished.values()),
        "measured_cost": False,
        "math_task_dag_unchanged": True,
        "logical_block_coverage": True,
        "mixed_pairing": True,
    }
    return cost_audit
