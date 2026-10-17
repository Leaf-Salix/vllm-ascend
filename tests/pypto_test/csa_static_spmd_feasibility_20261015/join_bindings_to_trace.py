# SPDX-License-Identifier: Apache-2.0
"""逐任务核对符号编排与真机 trace，绑定静态核序列需要的完整实参。"""

import argparse
import copy
import json
import math
from pathlib import Path


DTYPE_NAMES = {"FP32": "FLOAT32", "BF16": "BFLOAT16", "FP16": "FLOAT16",
               "INT8": "INT8", "INT32": "INT32", "INT64": "INT64"}


def physical_view(ref):
    shape, stride, offset = ref["shape"], ref["strides"], ref["offset"]
    if ref["layout"] == "NZ":
        if offset:
            raise ValueError("NZ GM slice 需要显式核对，不能按 ND 偏移转换")
        c0 = 32 if ref["dtype"] == "INT8" else 16
        m, n = shape[-2:]
        if m % 16 or n % c0:
            raise ValueError("NZ 固定权重维度没有对齐")
        shape = [math.prod(shape[:-2]), n // c0, m // 16, 16, c0]
        stride = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    return {"shape": shape, "strides": stride, "start_offset": offset,
            "dtype": DTYPE_NAMES[ref["dtype"]]}


def join(schedule, bindings, catalog):
    kernels = {k["name"]: k for k in catalog["kernels"]}
    # task_id 的数值不是提交次序。deps.json 原顺序由采集器保留；manifest 的
    # tasks 为便于阅读按 ID 排过序，因此以首次出现的原始 args/deps 顺序重建。
    dfx = Path(schedule["source"])
    original = json.loads((dfx / "deps.json").read_text())["tasks"]
    tasks = {t["task_id"]: t for t in schedule["tasks"]}
    traced = [tasks[t["task_id"]] for t in original if t["task_id"] in tasks]
    recorded = [t for t in bindings["tasks"] if t["kind"] == "compute"]
    if len(traced) != len(recorded):
        raise ValueError(f"计算调用数量不一致：{len(traced)} != {len(recorded)}")
    matches = {}
    checks = []
    for trace, call in zip(traced, recorded):
        logical = {kernels[name]["logical_task"] for name in trace["kernel_names"] if name}
        if logical != {call["logical_function"]} or trace["block_num"] != call["block_num"]:
            raise ValueError(f"提交次序/分片数量不一致：{trace['task_id']} {logical} / {call['logical_function']}")
        tensor_args = [a for a in call["arguments"] if isinstance(a["value"], dict)
                       and "allocation" in a["value"]]
        if len(tensor_args) != len(trace["args"]):
            raise ValueError(f"{call['logical_function']} Tensor 参数数量不一致")
        for arg, actual in zip(tensor_args, trace["args"]):
            expected = physical_view(arg["value"])
            actual_view = {k: int(actual[k]) if k == "start_offset" else actual[k] for k in expected}
            if expected != actual_view:
                raise ValueError(f"{call['logical_function']} / {arg['parameter']} 视图不一致："
                                 f"expected={expected}, observed={actual_view}")
        matches[trace["task_id"]] = call
        checks.append({"task_id": trace["task_id"], "call_ordinal": call["ordinal"],
                       "logical_function": call["logical_function"], "block_num": call["block_num"],
                       "tensor_arguments_checked": len(tensor_args), "status": "PASS"})

    prepared = copy.deepcopy(schedule)
    for core, sequence in prepared["per_core_sequences"].items():
        for member in sequence:
            if member.get("logical_block") is None:
                raise ValueError("完整调用绑定必须使用补齐 logical_block 的同轮 trace")
            call = matches[member["task"]]
            kernel = kernels[member["kernel_name"]]
            values = {a["parameter"]: a["value"] for a in call["arguments"]}
            values.update(original_block_idx=member["logical_block"],
                          original_block_num=member["logical_block_num"],
                          original_subblock_idx=1 if member["original_subslot"] == "AIV1" else 0,
                          original_subblock_num=2)
            missing = set(kernel["inline_parameter_order"]) - values.keys()
            if missing:
                raise ValueError(f"{member['kernel_name']} 缺少实参：{missing}")
            member.update(call_ordinal=call["ordinal"],
                          original_sync_start=bool(call["options"].get("sync_start", False)),
                          body_source_line=kernel["generated_line"],
                          body_parameter_names=kernel["inline_parameter_order"])
    # MIX 的配对由真实派发日志核对。纯 AIV 的 logical AIV0 可以派到任一物理
    # Vector；不能据 logical subslot 反推物理 lane。
    aic_count = sum(rows[0]["core_kind"] == "aic" for rows in prepared["per_core_sequences"].values())
    mixed = {}
    for sequence in prepared["per_core_sequences"].values():
        for m in sequence:
            task = tasks[m["task"]]
            if task["kernel_ids"][0] >= 0 and any(k >= 0 for k in task["kernel_ids"][1:]):
                mixed.setdefault((m["task"], m["logical_block"]), {})[m["original_subslot"]] = m["core"]
    for (task, block), group in mixed.items():
        aic = group["AIC"]
        for slot, offset in (("AIV0", 0), ("AIV1", 1)):
            if slot in group and group[slot] != aic_count + 2 * aic + offset:
                raise ValueError(f"MIX 同组配对不一致：{task}/{block} {group}")
    prepared.update(call_bindings={task: call for task, call in matches.items()},
                    temporary_allocations=bindings["allocations"],
                    root_arguments=bindings["root_arguments"],
                    binding_checks=checks,
                    mixed_logical_groups_checked=len(mixed),
                    all_call_bindings_verified=True,
                    body_bundle=catalog["body_bundle"],
                    scope="已绑定原分片、实参和缓冲的静态执行输入；尚非可执行的单 SPMD CSA")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = join(*(json.loads(p.read_text()) for p in (args.schedule, args.bindings, args.catalog)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"compute_calls": len(result["binding_checks"]),
                      "tensor_views_checked": sum(c["tensor_arguments_checked"] for c in result["binding_checks"]),
                      "mixed_groups_checked": result["mixed_logical_groups_checked"], "status": "PASS"}))


if __name__ == "__main__":
    main()
