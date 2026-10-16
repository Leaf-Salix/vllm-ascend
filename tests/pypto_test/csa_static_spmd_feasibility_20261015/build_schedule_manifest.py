# SPDX-License-Identifier: Apache-2.0
"""把一轮 CSA 派发转换为静态工作清单；本脚本不生成或执行静态 kernel。"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

from trace_audit import audit


DISPATCH = re.compile(
    r"Dispatched (?:pending|idle) (?P<slot>\w+) task (?P<task>\d+) "
    r"kernel_id=\[(?P<kernels>-?\d+,-?\d+,-?\d+)\] "
    r"block_idx=(?P<block>\d+)/total_blocks=(?P<blocks>\d+) "
    r"to core_offset=\d+ core_id=(?P<core>\d+) reg_task_id=(?P<seq>\d+)"
)
STAMP = re.compile(r"\d{4}-\d\d-\d\d-\d\d:\d\d:\d\d\.\d{3}\.\d{3}")
SLOTS = ("AIC", "AIV0", "AIV1")


def ancestor_sets(predecessors):
    """用位集计算 DAG 的完整先后关系，供依赖裁剪前后做严格比较。"""
    children = collections.defaultdict(list)
    degree = [len(p) for p in predecessors]
    for node, parents in enumerate(predecessors):
        for parent in parents:
            children[parent].append(node)
    ready = collections.deque(i for i, value in enumerate(degree) if value == 0)
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


def final_dispatches(directory):
    """只用于 capture_dispatch.sh：被捕获窗口必须是进程最后一次 CSA 执行。

    各次执行重新从 reg_task_id=1 开始。按每核序号回退分段，严格选最后一段，
    不搜索一个较早的“看起来能匹配”的执行。随后还会逐项核对任务和分片覆盖。
    CANN 日志可能异步落盘，所以不能用 Host 调用前后的文件大小切片。
    """
    lines = []
    for path in sorted(directory.rglob("*.log")):
        if not path.parent.name.startswith("device-"):
            continue
        timestamp = None
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            stamp = STAMP.search(line)
            if stamp is not None:
                timestamp = stamp.group()
            match = DISPATCH.search(line)
            if match:
                # CANN dlog 的格式是带时间戳的头一行，加下一行的带引号正文。
                if timestamp is None:
                    raise ValueError(f"派发日志缺少 CANN 时间戳：{path}:{number}")
                item = match.groupdict()
                item["slot"] = item["slot"].upper()
                item.update(core=int(item["core"]), seq=int(item["seq"]),
                            block=int(item["block"]), blocks=int(item["blocks"]),
                            kernels=[int(k) for k in item["kernels"].split(",")],
                            evidence=f"{path}:{number}")
                lines.append((timestamp, str(path), number, item))
    epochs = collections.defaultdict(list)
    for _, _, _, item in sorted(lines):
        segments = epochs[item["core"]]
        if not segments or item["seq"] <= segments[-1][-1]["seq"]:
            segments.append([])
        segments[-1].append(item)
    last = {}
    for core, segments in epochs.items():
        for item in segments[-1]:
            key = (core, item["seq"])
            if key in last:
                raise ValueError(f"最后执行的派发键重复：{key}")
            last[key] = item
    return last, {str(core): len(parts) for core, parts in sorted(epochs.items())}


def make_manifest(directory, dispatch_directory=None):
    raw = json.loads((directory / "chip_swimlane_records.json").read_text())
    deps = json.loads((directory / "deps.json").read_text())
    names = json.loads((directory / "name_map.json").read_text())["callable_id_to_name"]
    checked = audit(directory)
    if not checked["data_and_per_core_graph_acyclic"] or checked["dependency_timing_violations"]:
        raise ValueError("原任务依赖与核内顺序校验失败，不能生成静态清单")
    metadata = raw["metadata"]
    if len(metadata.get("run_boundaries", [])) != 1 or metadata.get("dropped_run_boundaries", 0):
        raise ValueError("要求完整的单次执行窗口")
    tasks = {t["task_id"]: t for t in deps["tasks"]}
    compute = {row["task"] for seq in checked["per_core_sequences"].values() for row in seq}
    parents = collections.defaultdict(set)
    for before, after in checked["compute_dependency_pairs_list"]:
        parents[after].add(before)
    mapping, epochs = ({}, {}) if dispatch_directory is None else final_dispatches(dispatch_directory)
    expected_keys = {(row[0], row[2]) for row in raw["aicore_tasks"]}
    if dispatch_directory is not None and set(mapping) != expected_keys:
        missing = sorted(expected_keys - set(mapping))[:8]
        extra = sorted(set(mapping) - expected_keys)[:8]
        raise ValueError(f"最后执行的日志与泳道成员不一致：missing={missing}, extra={extra}")

    per_core = {}
    members = collections.defaultdict(list)
    for core, rows in checked["per_core_sequences"].items():
        seqs = [r["dispatch_seq"] for r in rows]
        if seqs != list(range(1, len(rows) + 1)):
            raise ValueError(f"core {core} 存在缺失、重复或不支持的回绕序号：{seqs}")
        per_core[core] = []
        for row in rows:
            task = tasks[row["task"]]
            kind = metadata["core_types"][int(core)]
            candidate_slots = [i for i, k in enumerate(task["kernel_ids"])
                               if k >= 0 and (i == 0) == (kind == "aic")]
            item = dict(row, core_kind=kind, logical_block=None, logical_block_num=task["block_num"])
            item["possible_kernel_names"] = [names[str(task["kernel_ids"][i])] for i in candidate_slots]
            dispatch = mapping.get((int(core), row["dispatch_seq"]))
            if dispatch is not None:
                if dispatch["task"] != row["task"] or dispatch["kernels"] != task["kernel_ids"]:
                    raise ValueError(f"日志与泳道任务不一致：{core}/{row['dispatch_seq']}")
                if dispatch["blocks"] != task["block_num"] or not 0 <= dispatch["block"] < task["block_num"]:
                    raise ValueError(f"逻辑分片范围不一致：{dispatch}")
                slot = SLOTS.index(dispatch["slot"])
                if slot not in candidate_slots:
                    raise ValueError(f"核角色不一致：{dispatch}")
                item.update(logical_block=dispatch["block"], original_subslot=dispatch["slot"],
                            kernel_id=task["kernel_ids"][slot],
                            kernel_name=names[str(task["kernel_ids"][slot])],
                            dispatch_evidence=dispatch["evidence"])
            per_core[core].append(item)
            members[row["task"]].append(item)

    if mapping:
        # 每个逻辑分片必须恰好出现各个有效 kernel slot 一次；不能用执行顺序猜分片。
        for task_id, rows in members.items():
            task = tasks[task_id]
            actual = collections.Counter((r["original_subslot"], r["logical_block"]) for r in rows)
            expected = collections.Counter((SLOTS[i], b) for i, k in enumerate(task["kernel_ids"])
                                           if k >= 0 for b in range(task["block_num"]))
            if actual != expected:
                raise ValueError(f"task {task_id} 分片覆盖不完整或重复")

    # 先对原 task 完成依赖 + 每核程序顺序做传递约简。A→B→C 已经存在时，
    # 不再为 A→C 独立生成等待。这里严格保留原偏序，不改变分片、核映射或并行性。
    nodes = [row for rows in per_core.values() for row in rows]
    node_ids = {(str(row["core"]), row["dispatch_seq"]): i for i, row in enumerate(nodes)}
    original = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            original[i].add(node_ids[(str(row["core"]), row["dispatch_seq"] - 1)])
        for producer in parents[row["task"]]:
            for before in members[producer]:
                original[i].add(node_ids[(str(before["core"]), before["dispatch_seq"])])
    order, ancestors = ancestor_sets(original)
    position = {node: index for index, node in enumerate(order)}
    reduced = [set() for _ in nodes]
    for node, predecessors in enumerate(original):
        covered = 0
        for parent in sorted(predecessors, key=position.get, reverse=True):
            if not covered & (1 << parent):
                reduced[node].add(parent)
                covered |= ancestors[parent] | (1 << parent)

    # 把约简后的依赖翻译成“生产核至少执行到第几项”。同核先后隐含成立；
    # 一个消费核对同一生产核的多条要求取最大值，并省去之前已经等待过的进度。
    # 这是一份等待计划；release/acquire、缓存可见性和 MIX 内部事件仍须实现验证。
    published = collections.defaultdict(set)
    waits = 0
    for core, rows in per_core.items():
        observed = collections.defaultdict(int)
        for row in rows:
            frontier = collections.defaultdict(int)
            node = node_ids[(core, row["dispatch_seq"])]
            for parent in reduced[node]:
                before = nodes[parent]
                source = str(before["core"])
                if source == core:
                    if before["dispatch_seq"] >= row["dispatch_seq"]:
                        raise ValueError("同核依赖违反程序顺序")
                else:
                    frontier[source] = max(frontier[source], before["dispatch_seq"])
            required = {source: step for source, step in sorted(frontier.items(), key=lambda p: int(p[0]))
                        if step > observed[source]}
            row["wait_for_core_progress"] = required
            waits += len(required)
            for source, step in required.items():
                observed[source] = step
                published[source].add(step)
    for core, rows in per_core.items():
        for row in rows:
            row["publish_completion"] = row["dispatch_seq"] in published[core]
    proposed = [set() for _ in nodes]
    for i, row in enumerate(nodes):
        if row["dispatch_seq"] > 1:
            proposed[i].add(node_ids[(str(row["core"]), row["dispatch_seq"] - 1)])
        for source, step in row["wait_for_core_progress"].items():
            proposed[i].add(node_ids[(source, step)])
    _, proposed_ancestors = ancestor_sets(proposed)
    if proposed_ancestors != ancestors:
        raise ValueError("约简后出现依赖丢失或额外先后关系")

    task_manifest = []
    for task_id in sorted(compute, key=int):
        task = tasks[task_id]
        task_manifest.append(dict(task, kernel_names=[names.get(str(k)) if k >= 0 else None
                                                     for k in task["kernel_ids"]],
                                  predecessors=sorted(parents[task_id], key=int),
                                  member_count=len(members[task_id])))
    return {
        "schema_version": 1,
        "source": str(directory),
        "scope": "静态执行准备清单；不是已编译或已验证的单 SPMD CSA",
        "source_kind": "同轮 DEBUG 派发补采" if mapping else "已有泳道，逻辑分片尚未补齐",
        "summary": {k: checked[k] for k in ("physical_members", "compute_submissions",
                    "participating_cores", "compute_dependency_pairs", "data_and_per_core_graph_acyclic",
                    "dependency_timing_violations")},
        "logical_block_mapping_complete": bool(mapping),
        "dispatch_epochs_per_core": epochs,
        "wait_plan": {"remote_core_progress_requirements": waits,
                      "completion_publications": sum(len(v) for v in published.values()),
                      "original_member_edges": sum(map(len, original)),
                      "planned_member_edges": sum(map(len, proposed)),
                      "identical_dependency_partial_order": True,
                      "meaning": "核内顺序与原 task 依赖的等价表达，不包含 MIX 内部 pipe/event 握手",
                      "global_barriers_added": 0},
        "remaining_checks": [
            "从生成的 orchestration 对齐标量实参、视图绑定和 sync_start；deps.json 不包含完整调用 ABI。",
            "保留 MIX 内部同组 AIC/AIV 对应关系和 pipe/event 握手，检查事件编号复用。",
            "实现完成通知的存储可见性、重复 replay 的 flag 初始化/代际隔离。",
            "抽取纯计算体并显式传入原 block_idx/block_num；验证 UB/L1/L0 资源分配。",
            "尚未执行静态 kernel，也未测量或预测其性能。",
        ],
        "per_core_sequences": per_core,
        "tasks": task_manifest,
        "non_compute_tasks": [t for t in deps["tasks"] if t["task_id"] not in compute],
        "tensors": deps["tensors"],
        "original_edges": deps["edges"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfx", type=Path, required=True)
    parser.add_argument("--final-replay-device-logs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.final_replay_device_logs is not None:
        # 日志最后一轮只有在专门的补采驱动下才等于 DFX 窗口。
        report = json.loads((args.dfx.parent / "report.json").read_text())
        dfx = report.get("timing", {}).get("diagnostic_dfx", {})
        if (report.get("status") != "SCHEDULE_CAPTURED" or dfx.get("windows") != 1
                or not dfx.get("dispatch_debug_enabled")):
            raise ValueError("派发映射只接受单窗口补采驱动的最终 replay")
    manifest = make_manifest(args.dfx, args.final_replay_device_logs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "schedule_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    columns = ["core", "core_kind", "dispatch_seq", "task", "kernel_name", "logical_block",
               "logical_block_num", "original_subslot", "wait_for_core_progress", "publish_completion"]
    with (args.output / "per_core_schedule.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rows in manifest["per_core_sequences"].values():
            for row in rows:
                writer.writerow(dict(row, kernel_name=row.get("kernel_name", "|".join(row["possible_kernel_names"])),
                                     wait_for_core_progress=json.dumps(row["wait_for_core_progress"])))
    print(json.dumps(dict(manifest["summary"], mapping_complete=manifest["logical_block_mapping_complete"],
                          wait_plan=manifest["wait_plan"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
