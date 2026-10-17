# SPDX-License-Identifier: Apache-2.0
"""按当前形状在 CPU 上重新分配逻辑分片；代价估计不代表实测性能。

保留数学 task DAG、逻辑 block 集合与 MIX 核组配对。纯 AIC/AIV 可重新分配到
空闲核；原 trace 的偶然排队顺序不是数学依赖，不应永久固化为同一核的多次串行。
每个实际规格独立生成调度，不要求不同 batch 使用同一二进制。
"""

import collections
import copy
import math
import statistics
import json
from pathlib import Path


def earliest_fit(calendars, cores, release, durations):
    """在相关核的空隙中寻找共同起点，MIX 三个成员始终一起保留位置。"""
    start = release
    while True:
        conflict_ends = [end for core, duration in zip(cores, durations)
                         for begin, end in calendars[core]
                         if start < end - 1e-8 and start + duration > begin + 1e-8]
        if not conflict_ends:
            return start
        start = min(conflict_ends)


def rebalance(schedule, anchor, policy="critical_path", cost_profile=None, priority_weight=0.0,
              rank_bias=None, duration_scales=None):
    from shape_specialization import rebuild_waits

    if not math.isfinite(priority_weight) or not 0 <= priority_weight <= 10:
        raise ValueError("就绪时间与剩余关键路径的权重必须在0..10内")
    if priority_weight and policy != "earliest_ready":
        raise ValueError("额外关键路径权重只适用于earliest_ready")
    tasks = {task["task_id"]: task for task in schedule["tasks"]}
    rank_bias = dict(rank_bias or {})
    known_kernels = {next(n for n in task["kernel_names"] if n) for task in tasks.values()}
    if set(rank_bias) - known_kernels:
        raise ValueError("优先级偏置引用了不属于本档的数学核")
    if any(not math.isfinite(value) or abs(value) > 1000 for value in rank_bias.values()):
        raise ValueError("调度优先级偏置必须是-1000..1000内的有限数")
    examples, costs = {}, collections.defaultdict(list)
    for sequence in anchor["per_core_sequences"].values():
        for row in sequence:
            key = row["task"], row["original_subslot"]
            examples.setdefault(key, row)
            # 原 L1 50 MHz 时间戳，取各成员中位数；该估计仅用于分配优先级。
            costs[key].append((row["end"] - row["start"]) / 50)
    duration = {key: statistics.median(values) + 1.0 for key, values in costs.items()}
    cost_audit = None
    if cost_profile is not None:
        profile_path = Path(cost_profile)
        normalized = json.loads(profile_path.read_text())
        if isinstance(normalized, dict) and normalized.get("kind") in (
                "same_shape_dynamic_l1_cost", "same_shape_subtask_cost_sum"):
            current_shapes = {k: v["shape"] for k, v in schedule["root_arguments"].items()}
            if normalized["root_shapes"] != current_shapes:
                raise ValueError("原动态成本只允许用于完全相同的输入规格")
            entries = normalized["task_slots"]
            measured = {(r["task"], r["slot"]): r for r in entries}
            if len(measured) != len(entries) or set(measured) != set(duration):
                raise ValueError("原动态 L1 成本含重复项或未完整覆盖数学任务")
            for (task_id, slot), row in measured.items():
                index = ("AIC", "AIV0", "AIV1").index(slot)
                if row["kernel"] != tasks[task_id]["kernel_names"][index]:
                    raise ValueError("原动态成本与本规格核体不一致")
                if row["samples"] <= 0 or row["median_us"] <= 0:
                    raise ValueError("原动态成本必须来自有效 L1 样本")
            duration = {key: row["median_us"] + 1.0 for key, row in measured.items()}
            cost_audit = {"source": str(profile_path), "matched_task_slots": len(duration),
                          "same_shapes_checked": True, "source_kind": normalized["kind"],
                          "fused_cost_is_estimate": normalized["kind"] == "same_shape_subtask_cost_sum",
                          "used_only_for_offline_scheduling": True}
            cost_profile = None
    if cost_profile is not None:
        # 分档离线调优使用已采集的同规格阶段成本；线上不读取 profile，也
        # 不把等待时间计入计算成本，避免把旧调度的排队固化成数学负载。
        cost_profile = Path(cost_profile)
        probe_layout = json.loads((cost_profile.parent.parent / "layout.json").read_text())
        current_shapes = {k: v["shape"] for k, v in schedule["root_arguments"].items()}
        probe_schedule = json.loads(next((cost_profile.parent.parent / "specialization/bindings").glob(
            "*_prepared.json")).read_text())
        if current_shapes != {k: v["shape"] for k, v in probe_schedule["root_arguments"].items()}:
            raise ValueError("阶段成本只允许用于相同输入规格")
        slots = {(r["task"], r["core"], r["dispatch_seq"]): r["original_subslot"]
                 for batches in probe_layout["batches"].values() for batch in batches
                 for r in batch["members"]}
        measured = collections.defaultdict(list)
        for row in json.loads(cost_profile.read_text()):
            slot = slots[row["task"], row["core"], row["sequence"]]
            measured[row["task"], slot].append(row["math_and_fence_us"] + row["notify_us"])
        if set(measured) != set(duration):
            raise ValueError("阶段成本未完整覆盖当前数学 task 和物理成员类型")
        duration = {key: statistics.median(values) + 1.0 for key, values in measured.items()}
        cost_audit = {"source": str(cost_profile), "matched_task_slots": len(duration),
                      "same_shapes_checked": True, "wait_time_excluded": True,
                      "instrumented_cost_used_only_for_offline_scheduling": True}
    duration_scales = dict(duration_scales or {})
    if set(duration_scales) - known_kernels or any(
            not math.isfinite(scale) or not 0 < scale <= 8 for scale in duration_scales.values()):
        raise ValueError("列分块成本修正引用未知核或非法倍数")
    for (tid, slot), elapsed in list(duration.items()):
        kernel = tasks[tid]["kernel_names"][("AIC", "AIV0", "AIV1").index(slot)]
        duration[tid, slot] = elapsed * duration_scales.get(kernel, 1.0)
    children = collections.defaultdict(list)
    for task_id, task in tasks.items():
        for parent in task["predecessors"]:
            children[parent].append(task_id)
    ranks = {}

    def rank(task_id):
        if task_id not in ranks:
            task = tasks[task_id]
            slots = [slot for slot, name in zip(("AIC", "AIV0", "AIV1"), task["kernel_names"]) if name]
            capacity = 24 if task["kernel_names"][0] else 48
            own = max(duration[task_id, slot] for slot in slots) * math.ceil(task["block_num"] / capacity)
            ranks[task_id] = own + max((rank(child) for child in children[task_id]), default=0)
        return ranks[task_id]

    for task_id in tasks:
        rank(task_id)
    calendars = {core: [] for core in range(72)}
    queues = {str(core): [] for core in range(72)}
    finished = {}
    while len(finished) < len(tasks):
        ready = [tid for tid, task in tasks.items() if tid not in finished
                 and all(parent in finished for parent in task["predecessors"])]
        if not ready:
            raise ValueError("数学 task DAG 含环")
        def priority(task_id):
            task = tasks[task_id]
            if policy == "critical_path":
                start = 0
            elif policy == "earliest_ready":
                active = [slot for slot, name in zip(("AIC", "AIV0", "AIV1"), task["kernel_names"]) if name]
                release = max((finished[parent] for parent in task["predecessors"]), default=0)
                if task["kernel_names"][0] and any(task["kernel_names"][1:]):
                    candidates = [(g, 24 + 2 * g, 25 + 2 * g) for g in range(24)]
                elif task["kernel_names"][0]:
                    candidates = [(g,) for g in range(24)]
                else:
                    candidates = [(g,) for g in range(24, 72)]
                elapsed = [duration[task_id, slot] for slot in active]
                start = min(earliest_fit(calendars, cores, release, elapsed) for cores in candidates)
            else:
                raise ValueError("未知调度策略：" + policy)
            # 每档离线选择权重：0保持最早可开工，正值同时考虑后续关键路径。
            # 只影响任务放入核日历的先后，数学偏序和逻辑分片仍由下方完整核对。
            kernel = next(n for n in task["kernel_names"] if n)
            adjusted_rank = ranks[task_id] + rank_bias.get(kernel, 0.0)
            return (start - priority_weight * adjusted_rank, -adjusted_rank,
                    schedule["call_bindings"][task_id]["ordinal"])

        tid = min(ready, key=priority)
        task = tasks[tid]
        slots = [slot for slot, name in zip(("AIC", "AIV0", "AIV1"), task["kernel_names"]) if name]
        mixed = bool(task["kernel_names"][0] and any(task["kernel_names"][1:]))
        if mixed and slots != ["AIC", "AIV0", "AIV1"]:
            raise ValueError("当前 CSA 只验证了双 AIV 的 MIX 配对")
        release = max((finished[parent] for parent in task["predecessors"]), default=0)
        completion = release
        for block in range(task["block_num"]):
            if mixed:
                candidates = [(g, 24 + 2 * g, 25 + 2 * g) for g in range(24)]
            elif task["kernel_names"][0]:
                candidates = [(core,) for core in range(24)]
            else:
                if len(slots) != 1:
                    raise ValueError("不支持没有 AIC 的双成员任务")
                candidates = [(core,) for core in range(24, 72)]
            durations = [duration[tid, slot] for slot in slots]
            candidates = [(earliest_fit(calendars, cores, release, durations), cores)
                          for cores in candidates]
            start, cores = min(candidates)
            for slot, core, elapsed in zip(slots, cores, durations):
                end = start + elapsed
                row = copy.deepcopy(examples[tid, slot])
                for key in ("end", "dispatch_evidence", "wait_for_core_progress", "publish_completion"):
                    row.pop(key, None)
                row.update(core=core, logical_block=block, logical_block_num=task["block_num"],
                           logical_block_stride=task["block_num"], start=start * 50,
                           reference_end=end * 50, estimated_start_us=start, estimated_end_us=end)
                queues[str(core)].append(row)
                calendars[core].append((start, end))
                calendars[core].sort()
                completion = max(completion, end)
        finished[tid] = completion
    for core, sequence in queues.items():
        sequence.sort(key=lambda row: (row["estimated_start_us"], row["estimated_end_us"]))
        for seq, row in enumerate(sequence, 1):
            row["dispatch_seq"] = seq
    # 精确覆盖每一逻辑分片，且同一 MIX block 的 AIC/AIV 在同一硬件核组。
    coverage = collections.Counter((r["task"], r["original_subslot"], r["logical_block"])
                                   for rows in queues.values() for r in rows)
    expected = collections.Counter((tid, slot, block) for tid, task in tasks.items()
                                   for slot, name in zip(("AIC", "AIV0", "AIV1"), task["kernel_names"])
                                   if name for block in range(task["block_num"]))
    if coverage != expected:
        raise ValueError("重新分配后逻辑分片遗漏或重复")
    for tid, task in tasks.items():
        if not task["kernel_names"][0] or not any(task["kernel_names"][1:]):
            continue
        for block in range(task["block_num"]):
            members = {r["original_subslot"]: r["core"] for rows in queues.values() for r in rows
                       if r["task"] == tid and r["logical_block"] == block}
            if members["AIV0"] != 24 + 2 * members["AIC"] or members["AIV1"] != 25 + 2 * members["AIC"]:
                raise ValueError("MIX block 配对改变")
    schedule["per_core_sequences"] = queues
    schedule["wait_plan"] = rebuild_waits(queues, list(tasks.values()))
    schedule["source_kind"] = "shape_balanced_static_schedule"
    schedule["balanced_schedule"] = {
        "policy": policy,
        "priority_weight": priority_weight,
        "rank_bias": rank_bias,
        "unmeasured_duration_scales": duration_scales,
        "estimator": ("同规格实测成员中位数 + 1us" if cost_audit else
                      "参考档 L1 成员中位数 + 1us") + "；不是设备预测承诺",
        "measured_cost": cost_audit,
        "estimated_span_us": max(finished.values()),
        "math_task_dag_unchanged": True, "logical_block_coverage": True, "mixed_pairing": True,
        "old_per_core_order_preserved": False,
        "reason": "旧 trace 的资源排队不属于数学依赖，按当前形状重分配",
    }
    schedule["summary"]["physical_members"] = sum(map(len, queues.values()))
    return schedule["balanced_schedule"]
