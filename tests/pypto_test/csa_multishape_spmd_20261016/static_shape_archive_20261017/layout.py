# SPDX-License-Identifier: Apache-2.0
"""CSA 单 SPMD 的工作区及逐核执行表生成。

输入是已完成参数绑定和依赖核对的调度描述。只把逐核任务序列压成共同
源码批次；未参与某批次的核直接跳过，不引入全核屏障，不修改数学分片。
真实长度、位置和页表仍由核内读取 device Tensor。该模块不读测试文件，
不启动设备，也不把当前 runtime 的 ABI 容量假设写死为某个安装的数值。
"""

import collections
import copy
import math

ALIGNMENT = 512
PROGRESS_WORDS_PER_CORE = 64  # 各核独占 512 字节，不假定这就是硬件最小 cache line。
CORE_COUNT = 72
GENERATION_STRIDE = 1024
RECORD_WIDTH = 6  # 原 block、原 subslot、序号、等待表起点、等待数、是否发布。


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def batches_for_role(schedule, role, policy):
    """对每核字符串求公共超序列；只合并同时位于各核队首的同一计算调用。

    源码中的公共顺序不是跨核屏障：不属于该批的核直接跳过。原核内顺序、
    原 block 分配和等待目标逐项保留，不按时间重排或重新均分工作。
    """
    queues = {int(c): list(seq) for c, seq in schedule["per_core_sequences"].items() if seq[0]["core_kind"] == role}
    batches = []
    while any(queues.values()):
        heads = collections.defaultdict(list)
        for core, seq in queues.items():
            if seq:
                heads[(seq[0]["task"], seq[0]["kernel_name"])].append(core)

        def key(item):
            _, cores = item
            members = [queues[c][0] for c in cores]
            if policy == "ordinal":
                return (min(m["call_ordinal"] for m in members), -len(cores))
            if policy == "earliest":
                return (min(m["start"] for m in members), -len(cores))
            return (-len(cores), min(m["call_ordinal"] for m in members))

        (task, kernel), cores = min(heads.items(), key=key)
        batches.append({"task": task, "kernel": kernel, "members": [queues[c].pop(0) for c in sorted(cores)]})
    return batches


def prepare(schedule, catalog):
    if not schedule["all_call_bindings_verified"]:
        raise ValueError("未完成原调用绑定，不能生成 SPMD 入口")
    bodies = {k["name"]: k for k in catalog["kernels"]}
    batches, choices = {}, {}
    for role in ("aic", "aiv"):
        candidates = {p: batches_for_role(schedule, role, p) for p in ("ordinal", "earliest", "most_heads")}
        selected = min(candidates, key=lambda p: len(candidates[p]))
        batches[role] = candidates[selected]
        choices[role] = {"selected": selected, "counts": {k: len(v) for k, v in candidates.items()}}

    # 重建逐核轨迹做等价检查，而非仅统计批次数。
    replay = collections.defaultdict(list)
    for role_batches in batches.values():
        for batch in role_batches:
            for member in batch["members"]:
                replay[str(member["core"])].append(member)
    if dict(replay) != schedule["per_core_sequences"]:
        raise ValueError("批次生成改变了逐核轨迹")

    # 先不复用任何临时 allocation。这样不需要为节约 GM 额外引入等待，
    # 不会误把时间线上的偶然先后当作可复用生命周期。
    records, waits = [], []
    for role_batches in batches.values():
        for batch in role_batches:
            batch["record_offset_words"] = len(records)
            by_core = {m["core"]: m for m in batch["members"]}
            for core in range(CORE_COUNT):
                member = by_core.get(core)
                if member is None:
                    records.extend([-1, 0, 0, 0, 0, 0])
                    continue
                start = len(waits) // 2
                for other, sequence in sorted(member["wait_for_core_progress"].items(), key=lambda x: int(x[0])):
                    waits.extend([int(other), sequence])
                records.extend(
                    [
                        member["logical_block"],
                        1 if member["original_subslot"] == "AIV1" else 0,
                        member["dispatch_seq"],
                        start,
                        len(member["wait_for_core_progress"]),
                        int(member["publish_completion"]),
                    ]
                )
    if max(m["dispatch_seq"] for seq in replay.values() for m in seq) >= GENERATION_STRIDE:
        raise ValueError("代际步长不足以容纳本轮最大核内序号")
    offset = 0
    sections = {}
    for name, count in (
        ("progress", CORE_COUNT * PROGRESS_WORDS_PER_CORE),
        ("records", len(records)),
        ("waits", len(waits)),
    ):
        sections[name] = {"offset_bytes": offset, "bytes": count * 8, "shape": [count], "dtype": "INT64"}
        offset = align(offset + count * 8)
    temporaries = {}
    for allocation in schedule["temporary_allocations"]:
        item = copy.deepcopy(allocation)
        if item["strides"] != [math.prod(item["shape"][i + 1 :]) for i in range(len(item["shape"]))]:
            raise ValueError("工作区只接收原连续 allocation，不能猜测带 stride 的实际大小")
        item["offset_bytes"] = offset
        temporaries[item["allocation"]] = item
        offset = align(offset + item["bytes"])

    events = []
    for task, call in schedule["call_bindings"].items():
        names = {m["kernel_name"] for seq in replay.values() for m in seq if m["task"] == task}
        for name in sorted(names):
            if bodies[name]["system_operations"]:
                arguments = {a["parameter"]: a["value"] for a in call["arguments"]}
                operations = copy.deepcopy(bodies[name]["system_operations"])
                for op in operations:
                    if op["operation"] == "pl.system.set_ffts":
                        op["bound_workspace"] = arguments[op["arguments"][0]]
                events.append({"task": task, "kernel": name, "call_ordinal": call["ordinal"], "operations": operations})
    roots = schedule["root_arguments"]
    # 这里只统计真实描述符。运行时参数容量必须由调用方提供并验证，
    # 不能把早期 32 槽或某个独立 256 槽安装的能力写死在通用生成器里。
    abi = {
        "runtime_core_tensor_limit": None,
        "original_root_tensors": len(roots),
        "original_temporary_allocations": len(temporaries),
        "hypothetical_root_tensors_with_single_workspace": len(roots) + 1,
        "leaf_tensor_count_source": "codegen.generate: _entry_contract.leaf_tensor_count",
        "fits_current_runtime": None,
        "validation": "待 compiler 按调用方明确提供的 runtime 容量核对",
    }
    return {
        "schema_version": 1,
        "source_schedule": schedule["source"],
        "scope": "生成当前形状的 SPMD 工作区与依赖表；不代表设备验证或性能结论",
        "per_core_order_mapping_and_waits_exact": True,
        "shared_source_order_is_not_a_barrier": True,
        "batch_choices": choices,
        "batches": batches,
        "workspace": {
            "bytes": offset,
            "alignment": ALIGNMENT,
            "sections": sections,
            "temporaries": temporaries,
            "reuse": "none",
            "records": records,
            "waits": waits,
        },
        "protocol": {
            "generation_stride": GENERATION_STRIDE,
            "progress_words_per_core": PROGRESS_WORDS_PER_CORE,
            "record_width": RECORD_WIDTH,
            "core_count": CORE_COUNT,
            "initial_progress": 0,
            "workspace_private_per_concurrent_invocation": True,
            "global_barrier_added": False,
            "device_visibility_and_replay_validated": False,
        },
        "existing_internal_sync": events,
        "abi": abi,
    }
