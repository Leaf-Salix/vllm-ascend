# SPDX-License-Identifier: Apache-2.0
"""CPU 核对静态表覆盖、工作区边界及跨重放的进度代际；不是硬件同步测试。"""

import argparse
import collections
import json
import math
import random
from pathlib import Path

from bind_orchestration import DTYPE_BYTES


def validate(schedule, layout):
    workspace = layout["workspace"]
    records, waits = workspace["records"], workspace["waits"]
    width = layout["protocol"]["record_width"]
    reconstructed = collections.defaultdict(list)
    for role, batches in layout["batches"].items():
        for batch in batches:
            for core in range(72):
                start = batch["record_offset_words"] + core * width
                block, subslot, sequence, base, count, publish = records[start:start + width]
                if block < 0:
                    continue
                if (core < 24) != (role == "aic"):
                    raise ValueError("记录被绑定到了错误的 AIC/AIV 角色")
                dependencies = {str(waits[(base + i) * 2]): waits[(base + i) * 2 + 1] for i in range(count)}
                reconstructed[str(core)].append({"task": batch["task"], "kernel_name": batch["kernel"],
                    "logical_block": block, "original_subslot": "AIC" if role == "aic" else f"AIV{subslot}",
                    "dispatch_seq": sequence, "wait_for_core_progress": dependencies,
                    "publish_completion": bool(publish)})
    for core, seq in schedule["per_core_sequences"].items():
        reconstructed_seq = reconstructed[core]
        fields = reconstructed_seq[0].keys()
        expected = [{k: member[k] for k in fields} for member in seq]
        if expected != reconstructed_seq:
            raise ValueError(f"静态记录不能完整重建核 {core} 的分片、顺序和等待")
    regions = []
    for name, value in {**workspace["sections"], **workspace["temporaries"]}.items():
        begin, end = value["offset_bytes"], value["offset_bytes"] + value["bytes"]
        if begin % workspace["alignment"] or begin < 0 or end > workspace["bytes"]:
            raise ValueError(f"工作区范围或对齐无效：{name}")
        regions.append((begin, end, name))
    regions.sort()
    if any(a[1] > b[0] for a, b in zip(regions, regions[1:])):
        raise ValueError("工作区 allocation 重叠")
    refs_checked = 0
    for call in schedule["call_bindings"].values():
        for arg in call["arguments"]:
            ref = arg["value"]
            if not isinstance(ref, dict) or "allocation" not in ref:
                continue
            if ref["allocation"].startswith("argument:"):
                root = schedule["root_arguments"][ref["allocation"][9:]]
                limit = math.prod(root["shape"])
            else:
                root = workspace["temporaries"][ref["allocation"]]
                limit = root["bytes"] // DTYPE_BYTES[root["dtype"]]
            if ref["dtype"] != root["dtype"]:
                raise ValueError("Tensor 视图擅自改变 dtype")
            last = ref["offset"] + sum((n - 1) * stride for n, stride in zip(ref["shape"], ref["strides"]))
            if ref["offset"] < 0 or last >= limit:
                raise ValueError(f"Tensor 视图越界：{arg['parameter']}")
            refs_checked += 1

    # 只验证逻辑进度与代际，假定原子读写以及 release/acquire 正确。
    # 不能以这项模型结果代替 NPU cache coherence、pipe/event 死锁验证。
    repetitions = []
    stride = layout["protocol"]["generation_stride"]
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        flags = {c: 0 for c in reconstructed}
        for generation in range(1, 4):
            position = {c: -1 for c in reconstructed}
            steps = 0
            while any(position[c] < len(seq) for c, seq in reconstructed.items()):
                ready = []
                for core, seq in reconstructed.items():
                    index = position[core]
                    if index == -1:
                        ready.append(core)
                    elif index < len(seq) and all(flags[other] >= generation * stride + value
                                                  for other, value in seq[index]["wait_for_core_progress"].items()):
                        ready.append(core)
                if not ready:
                    raise ValueError(f"进度协议模型无可前进的核：seed={seed}, generation={generation}")
                core = rng.choice(ready)
                index = position[core]
                if index == -1:
                    own_generation = flags[core] // stride + 1
                    if own_generation != generation:
                        raise ValueError("重放代际失配")
                    flags[core] = own_generation * stride
                else:
                    item = reconstructed[core][index]
                    if item["publish_completion"]:
                        flags[core] = generation * stride + item["dispatch_seq"]
                position[core] += 1
                steps += 1
            repetitions.append({"seed": seed, "generation": generation, "steps": steps, "status": "PASS"})
    return {"table_exact_reconstruction": "PASS", "workspace_regions_disjoint": "PASS",
            "tensor_views_in_bounds": refs_checked, "progress_generation_model": repetitions,
            "scope": "CPU 逻辑模型；不证明真实设备的原子性、可见性、内部事件收敛或性能",
            "device_execution": False, "hardware_protocol_validated": False,
            "hypothetical_root_tensor_arguments": layout["abi"]["hypothetical_root_tensors_with_single_workspace"],
            "actual_leaf_abi_source": "entry.bindings.json:_entry_contract",
            "current_runtime_abi_supported": layout["abi"]["fits_current_runtime"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.prepared.read_text()), json.loads(args.layout.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "progress_generation_model"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
