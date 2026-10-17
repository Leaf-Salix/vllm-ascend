# SPDX-License-Identifier: Apache-2.0
"""读取已完成的整层结果，核对 HC 有效消费者通知在真实设备上的发布集合。

不导入 torch/PyPTO，不重新编译或执行 NPU。原整层数值、图、保护区与
代际验收必须先通过；本脚本只增加四个阶段的信号检查，不能替代它们。
观察对象是首次清零后、固定 T 绑定下 A→B→A 三轮原始 signal 快照；
保持不活跃槽的上轮值，不假设运行时会清零。本记录不证明跨 shape 图重放。
"""

import argparse
import json
from pathlib import Path


def inspect(report, phases):
    if not report.get("passed") or report.get("stage") != "complete":
        raise ValueError("必须先完成原整层验收，不能只凭信号将失败作业标为通过")
    tokens = report["shapes"]["x"][0]
    consumer_reduce = report["policy"]["HC_CONSUMER_REDUCE"]
    launches = report["launches"]
    if not launches:
        raise ValueError("没有真实设备信号快照")
    previous = {name: [0] * 72 for name in ("RAW", "GATES", "NORM", "COMB")}
    rows = []
    # 从消费者实际访问的行块建立集合；不调用候选的publish谓词。
    gate_blocks = {row // 8 for row in range(tokens)}
    raw_blocks = {row // 16 for row in range(tokens)}
    owners = {
        "RAW": set() if consumer_reduce else {24 + block % 48 for block in raw_blocks},
        "GATES": {24 + block % 48 for block in gate_blocks},
        "NORM": {24 + block % 48 for block in gate_blocks},
        "COMB": {24 + block % 48 for block in gate_blocks},
    }
    for launch in launches:
        if not launch["passed"]:
            raise AssertionError("原整层launch未通过")
        generation = launch["generation"]
        observed = launch["observed_phase_slots"]
        if len(observed) != 60 or any(len(row) != 72 for row in observed):
            raise AssertionError("本检查仅适用于原60阶段、72核协议")
        if observed[0] != [generation] * 72:
            raise AssertionError("72核begin代际发生改变")
        for name, active in owners.items():
            expected = previous[name].copy()
            for core in active:
                expected[core] = generation
            if observed[phases[name]] != expected:
                raise AssertionError((generation, name, observed[phases[name]], expected))
            previous[name] = expected
            rows.append({
                "generation": generation,
                "phase": name,
                "writers": sorted(active),
                "inactive_values_preserved": True,
            })
    prior_count = 48 * (3 if consumer_reduce else 4)
    return {
        "passed": True,
        "scope": "首次清零、固定T、已完成整层作业的四阶段真实设备发布集合；不是性能测试",
        "new_device_execution": False,
        "tokens": tokens,
        "launches": len(launches),
        "prior_publications_per_call": prior_count,
        "observed_publications_per_call": sum(map(len, owners.values())),
        "skipped_publications_per_call": prior_count - sum(map(len, owners.values())),
        "checks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", type=Path,
                        default=Path(__file__).resolve().parents[2] / "source_interface.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("结果路径必须全新，不覆盖历史")
    phases = json.loads(args.interface.read_text())["phases"]
    result = inspect(json.loads(args.report.read_text()), phases)
    result["source_report"] = str(args.report.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
