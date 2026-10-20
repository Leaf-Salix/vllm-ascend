# SPDX-License-Identifier: Apache-2.0
"""回读完整用例已经保存的信号，检查按输出组通知的真实设备发布者。

仅检查本套固定T绑定、首次清零信号区的eager快照。按全局逻辑任务枚举
owner，不复用设备首项公式；不新增设备调用，不替代完整输出及保护检查。
"""

import argparse
import json
from pathlib import Path


def validate(report):
    assert report["policy"]["OUTPUT_GROUP_PIPELINE"]
    assert report["launches"], "尚未获得完整设备快照"
    tokens = report["shapes"]["x"][0]
    oa_per_group = ((tokens + 127) // 128) * (1024 // report["policy"]["OA_COLS"])
    quant_per_group = (tokens + 31) // 32
    results = []
    for launch in report["launches"]:
        slots = launch["observed_phase_slots"]
        generation = launch["generation"]
        assert len(slots) == 60 and all(len(row) == 72 for row in slots)
        assert not any(slots[40]) and not any(slots[41]), "组流水不应发布旧整阶段槽"
        for kind, base, workers, core_base, per_group in (
            ("OA", 44, 24, 0, oa_per_group),
            ("O_QUANT", 52, 48, 24, quant_per_group),
        ):
            for group in range(8):
                owners = {core_base + n % workers for n in range(group * per_group, (group + 1) * per_group)}
                expected = [generation if core in owners else 0 for core in range(72)]
                assert slots[base + group] == expected, (generation, kind, group, slots[base + group], expected)
                results.append(dict(generation=generation, stage=kind, group=group, owners=sorted(owners)))
    return dict(passed=True, scope="已有固定T设备快照的输出组phase owner检查", new_device_execution=False, rows=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.result.read_text()))
    target = args.result.parent / "output_group_signal_check.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=True, checked=len(result["rows"]), output=str(target)), ensure_ascii=False))


if __name__ == "__main__":
    main()
