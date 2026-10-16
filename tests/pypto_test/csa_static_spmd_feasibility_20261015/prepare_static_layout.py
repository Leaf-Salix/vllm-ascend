# SPDX-License-Identifier: Apache-2.0
"""为七档静态 CSA 准备核序列、GM 工作区及入口容量清单；不运行 NPU。"""

import argparse
import collections
import copy
import json
import math
from pathlib import Path


# 这里只分类层初始化后不变的权重；不把 RoPE、slot、长度等运行时输入冒充权重。
WEIGHTS = frozenset("""hc_attn_fn hc_attn_scale hc_attn_base attn_norm_w wq_a wq_b
wq_b_scale wkv gamma_cq gamma_ckv cmp_wkv cmp_wgate cmp_ape cmp_norm_w idx_wq_b
idx_wq_b_scale weights_proj hadamard_idx inner_wkv inner_wgate inner_ape
inner_norm_w attn_sink wo_a wo_b wo_b_scale""".split())
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
    queues = {int(c): list(seq) for c, seq in schedule["per_core_sequences"].items()
              if seq[0]["core_kind"] == role}
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
        batches.append({"task": task, "kernel": kernel,
                        "members": [queues[c].pop(0) for c in sorted(cores)]})
    return batches


def prepare(schedule, catalog):
    if not schedule["all_call_bindings_verified"]:
        raise ValueError("未完成原调用绑定，不能生成静态入口")
    bodies = {k["name"]: k for k in catalog["kernels"]}
    batches, choices = {}, {}
    for role in ("aic", "aiv"):
        candidates = {p: batches_for_role(schedule, role, p)
                      for p in ("ordinal", "earliest", "most_heads")}
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
                records.extend([member["logical_block"],
                                1 if member["original_subslot"] == "AIV1" else 0,
                                member["dispatch_seq"], start,
                                len(member["wait_for_core_progress"]), int(member["publish_completion"])])
    if max(m["dispatch_seq"] for seq in replay.values() for m in seq) >= GENERATION_STRIDE:
        raise ValueError("代际步长不足以容纳本轮最大核内序号")
    offset = 0
    sections = {}
    for name, count in (("progress", CORE_COUNT * PROGRESS_WORDS_PER_CORE),
                        ("records", len(records)), ("waits", len(waits))):
        sections[name] = {"offset_bytes": offset, "bytes": count * 8,
                          "shape": [count], "dtype": "INT64"}
        offset = align(offset + count * 8)
    temporaries = {}
    for allocation in schedule["temporary_allocations"]:
        item = copy.deepcopy(allocation)
        if item["strides"] != [math.prod(item["shape"][i + 1:]) for i in range(len(item["shape"]))]:
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
                events.append({"task": task, "kernel": name, "call_ordinal": call["ordinal"],
                               "operations": operations})
    roots = schedule["root_arguments"]
    nz_weights = [name for name in WEIGHTS if roots[name]["layout"] == "NZ"]
    # 此处只列候选容量，不执行权重复制，也不更改任何入口权重格式。
    abi = {"runtime_core_tensor_limit": 32, "original_root_tensors": len(roots),
           "original_temporary_allocations": len(temporaries),
           "hypothetical_root_tensors_with_single_workspace": len(roots) + 1,
           "leaf_tensor_count_source": "entry.bindings.json:_entry_contract.leaf_tensor_count",
           "fits_current_runtime": False,
           "source": "simpler/src/common/task_interface/arg_direction.h:29",
           "immutable_weights": sorted(WEIGHTS), "nz_weights": sorted(nz_weights),
           "hypothetical_all_weight_packing_tensors": len(roots) - len(WEIGHTS) + 2,
           "nd_only_weight_packing_tensors": len(roots) - len(WEIGHTS) + len(nz_weights) + 2,
           "packing_status": "NOT_IMPLEMENTED_NOT_ASSUMED_SUPPORTED",
           "packing_constraint": "现有 tensor.view 不能产生 NZ，reinterpret_view 不接受 NZ；"
                                 "不能把四份 NZ 权重标注成 ND 冒充零拷贝等价。"}
    return {"schema_version": 1, "source_schedule": schedule["source"],
            "scope": "静态源代码与协议准备；未执行静态 CSA，不包含性能结论",
            "per_core_order_mapping_and_waits_exact": True,
            "shared_source_order_is_not_a_barrier": True,
            "batch_choices": choices, "batches": batches,
            "workspace": {"bytes": offset, "alignment": ALIGNMENT, "sections": sections,
                          "temporaries": temporaries, "reuse": "none", "records": records, "waits": waits},
            "protocol": {"generation_stride": GENERATION_STRIDE,
                         "progress_words_per_core": PROGRESS_WORDS_PER_CORE,
                         "record_width": RECORD_WIDTH, "core_count": CORE_COUNT,
                         "initial_progress": 0, "workspace_private_per_concurrent_invocation": True,
                         "global_barrier_added": False,
                         "device_visibility_and_replay_validated": False},
            "existing_internal_sync": events, "abi": abi}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.prepared.read_text()), json.loads(args.catalog.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"batches": {r: len(v) for r, v in result["batches"].items()},
                      "workspace_bytes": result["workspace"]["bytes"],
                      "hypothetical_root_tensor_arguments": result["abi"]["hypothetical_root_tensors_with_single_workspace"],
                      "runtime_abi_supported": result["abi"]["fits_current_runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
