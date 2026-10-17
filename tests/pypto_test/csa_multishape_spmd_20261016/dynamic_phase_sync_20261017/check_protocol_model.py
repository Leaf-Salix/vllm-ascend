# SPDX-License-Identifier: Apache-2.0
"""纯 CPU 推进模型：检查阶段/代际逻辑，不模拟 DCache、流水或 NPU。

与 check_cpu.py 相同的两阶段生产/消费图，使用立即可见的抽象 GM。
调度器从所有可推进核随机选择一步，不使用时间、不调用 PyPTO、不编译。
因此 PASS 只说明本模型中没有提前读 payload 或依赖死锁；不证明硬件同步。
"""

import argparse
import json
import random
from pathlib import Path


def plans(groups, rows, skip_empty_publish=False):
    """生成与现有探针对应的核内顺序；可注入空核漏发布负例。"""
    result = []
    for core in range(groups):
        owned = [row for row in range(rows) if (row // 16) % groups == core]
        result.append([
            ("begin",), ("produce", owned), ("publish", 1),
            ("wait", 2, groups + ((core + 1) % groups) * 2),
        ])
    for worker in range(2 * groups):
        events = [("begin",)]
        has_work = False
        for tile in range(worker, (rows + 7) // 8, 2 * groups):
            covered = list(range(tile * 8, min(rows, tile * 8 + 8)))
            owners = sorted({(row // 16) % groups for row in covered})
            events.extend(("wait", 1, owner) for owner in owners)
            events.append(("consume", covered))
            has_work = True
        if has_work or not skip_empty_publish:
            events.append(("publish", 2))
        result.append(events)
    return result


def run_invocation(flags, groups, rows, invocation, seed, skip_empty_publish=False):
    rng = random.Random(seed)
    operations = plans(groups, rows, skip_empty_publish)
    participants = groups * 3
    cursor = [0] * participants
    generation = [None] * participants
    produced = {}
    consumed = set()
    steps = 0
    recent = []
    while any(cursor[c] < len(operations[c]) for c in range(participants)):
        ready = []
        for core, events in enumerate(operations):
            if cursor[core] == len(events):
                continue
            event = events[cursor[core]]
            if event[0] != "wait" or flags.get(event[1] * participants + event[2], 0) >= generation[core]:
                ready.append(core)
        if not ready:
            return {"status": "deadlock", "steps": steps, "generations": generation, "recent": recent}
        core = rng.choice(ready)
        event = operations[core][cursor[core]]
        kind = event[0]
        if kind == "begin":
            generation[core] = flags.get(core, 0) + 1
            flags[core] = generation[core]
        elif kind == "produce":
            produced.update((row, invocation) for row in event[1])
        elif kind == "publish":
            flags[event[1] * participants + core] = generation[core]
        elif kind == "consume":
            missing = [row for row in event[1] if produced.get(row) != invocation]
            if missing:
                return {"status": "premature_payload_read", "core": core, "rows": missing,
                        "generations": generation, "steps": steps, "recent": recent}
            consumed.update(event[1])
        cursor[core] += 1
        steps += 1
        recent = (recent + [{"core": core, "operation": kind}])[-12:]
    if consumed != set(range(rows)):
        return {"status": "missing_output_rows", "consumed": sorted(consumed), "rows": rows}
    return {"status": "complete", "steps": steps, "generations": generation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows_by_round = [0, 1, 7, 8, 15, 16, 17, 95, 96, 97, 168, 385, 769, 0]
    total_runs = total_steps = 0
    for groups in (1, 2, 3, 24):
        for seed in range(32):
            flags = {}
            for invocation, rows in enumerate(rows_by_round, 1):
                outcome = run_invocation(flags, groups, rows, invocation, seed * 100 + invocation)
                assert outcome["status"] == "complete", (groups, seed, invocation, outcome)
                assert set(outcome["generations"]) == {invocation}, outcome
                total_runs += 1
                total_steps += outcome["steps"]

    omitted = run_invocation({}, 3, 1, 1, 0, skip_empty_publish=True)
    assert omitted["status"] == "deadlock", omitted

    # 故意违反“同一 workspace 的 G 不变”：6组 -> 3组 -> 6组。
    # 第3轮开始前的 stage0 已混入上轮另一个布局的 stage1，不能再代表统一调用代际。
    flags = {}
    for invocation, groups in enumerate((6, 3), 1):
        outcome = run_invocation(flags, groups, 96, invocation, invocation)
        assert outcome["status"] == "complete", outcome
    next_generations = [flags.get(core, 0) + 1 for core in range(18)]
    assert len(set(next_generations)) > 1, next_generations
    outcomes = {}
    example = None
    for seed in range(256):
        outcome = run_invocation(dict(flags), 6, 96, 3, seed)
        outcomes[outcome["status"]] = outcomes.get(outcome["status"], 0) + 1
        if outcome["status"] != "complete" and example is None:
            example = {"seed": seed, **outcome}
    assert example is not None, outcomes

    # 阶段号是整个 producer 阶段，不是可无ACK重复使用的 tile槽。
    repeated_phase = {"expected_generation": 1, "after_first_tile_flag": 1,
                      "second_tile_published": False, "second_wait_passes": 1 >= 1}
    assert repeated_phase["second_wait_passes"] and not repeated_phase["second_tile_published"]
    report = {
        "scope": "立即可见GM的纯CPU协议逻辑模型；不模拟缓存/流水/编译器/硬件",
        "device_execution": False, "recompilation": False, "hardware_correctness_proven": False,
        "fixed_layout": {"passed": True, "invocations": total_runs, "event_steps": total_steps,
                         "groups": [1, 2, 3, 24], "seeds": 32, "rows_by_round": rows_by_round},
        "negative_omitted_empty_publish": omitted,
        "negative_changed_layout": {"groups_by_round": [6, 3, 6], "next_generations": next_generations,
                                    "outcomes_over_256_seeds": outcomes, "first_failure": example},
        "negative_reused_phase_for_second_tile": repeated_phase,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
