# SPDX-License-Identifier: Apache-2.0
"""读取已完成的动态 PTO L1，导出同规格离线调度成本；不启动设备测量。"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def export(case, output):
    case = Path(case)
    result = json.loads((case / "result.json").read_text())
    if not result.get("passed") or result.get("native_forward_executed"):
        raise ValueError("必须使用已通过的 PTO 对照记录")
    binding_files = list((case / "specialization/bindings").glob("*_prepared.json"))
    if len(binding_files) != 1:
        raise ValueError("当前结果不能唯一定位绑定清单")
    schedule = json.loads(binding_files[0].read_text())
    tasks = {row["task_id"]: row for row in schedule["tasks"]}
    costs, paths = collections.defaultdict(list), []
    for window in result["l1_windows"]:
        if window["side"] != "dynamic_pto":
            continue
        path = Path(window["raw"])
        trace = json.loads(path.read_text())
        hz = trace["metadata"]["clock_freq_hz"]
        if hz <= 0:
            raise ValueError("L1 时钟频率无效")
        paths.append(str(path))
        for core, tid, sequence, start, end, _ in trace["aicore_tasks"]:
            # 六列里的第三列是核内派发序号，不能当 kernel_id 使用。
            task = tasks[str(tid)]
            names = task["kernel_names"]
            if core < 24:
                slot = "AIC"
            elif names[0] is not None:
                slot = ("AIV0", "AIV1")[(core - 24) % 2]
            else:
                active = [s for s, n in zip(("AIV0", "AIV1"), names[1:]) if n]
                if len(active) != 1:
                    raise ValueError("无法唯一识别纯 Vector 成员")
                slot = active[0]
            if not names[("AIC", "AIV0", "AIV1").index(slot)] or end <= start:
                raise ValueError("L1 成员与核体绑定不一致")
            costs[str(tid), slot].append((end - start) * 1e6 / hz)
    expected = {(tid, slot) for tid, task in tasks.items()
                for slot, name in zip(("AIC", "AIV0", "AIV1"), task["kernel_names"]) if name}
    if set(costs) != expected:
        raise ValueError("动态 L1 未完整覆盖当前数学任务")
    exported = {
        "kind": "same_shape_dynamic_l1_cost", "source_case": str(case), "raw_paths": paths,
        "root_shapes": {k: v["shape"] for k, v in schedule["root_arguments"].items()},
        "task_slots": [{"task": tid, "slot": slot,
                        "kernel": tasks[tid]["kernel_names"][("AIC", "AIV0", "AIV1").index(slot)],
                        "samples": len(values), "median_us": statistics.median(values)}
                       for (tid, slot), values in sorted(costs.items())],
        "note": "只作相同规格的离线分核成本；包含成员内局部等待，不能作为新调度的性能承诺。",
    }
    Path(output).write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n")
    return {"members": len(costs), "windows": len(paths), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.case, args.output), ensure_ascii=False))
