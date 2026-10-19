# SPDX-License-Identifier: Apache-2.0
"""用整数模型核对设备分片覆盖、正反映射及当前诊断的逐核实际步数。

该检查针对调度所有权，不能代替矩阵精度、缓存可见性、图重放或性能。
只读已有Score诊断，不重新运行NPU。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORES = 24
TILE = 1024
LEAF = 8192
MAX_LEAVES = 32
DIVISORS = (1, 2, 3, 4, 6, 8, 12, 24)


def plan(history, groups, revised=False):
    tiles = (history + TILE - 1) // TILE
    leaves = max((history + LEAF - 1) // LEAF, 1)
    width, extra, multiple = 8, 0, 1
    if 3 * groups == 2 * CORES or 3 * groups == CORES:
        multiple = 3
    elif 6 * groups == CORES:
        multiple = 6
    if multiple > 1:
        balanced = (leaves + multiple - 1) // multiple * multiple
        if balanced <= MAX_LEAVES:
            leaves, width, extra = balanced, max(tiles // balanced, 1), tiles % balanced
    if revised and groups >= CORES and leaves <= CORES:
        aligned = next(n for n in DIVISORS if n >= leaves)
        if aligned <= tiles and aligned <= MAX_LEAVES:
            leaves, width, extra = aligned, tiles // aligned, tiles % aligned
    return leaves, width, extra


def decode(item, groups, leaves, revised=False):
    if groups < CORES:
        return item % groups, item // groups
    group, leaf = divmod(item, leaves)
    if revised and leaves <= CORES and CORES % leaves == 0:
        rotation = group // (CORES // leaves)
        leaf = (leaf + rotation) % leaves
    return group, leaf


def encode(group, leaf, groups, leaves, revised=False):
    if groups < CORES:
        return leaf * groups + group
    physical_leaf = leaf
    if revised and leaves <= CORES and CORES % leaves == 0:
        rotation = group // (CORES // leaves)
        physical_leaf = (leaf + leaves - rotation % leaves) % leaves
    return group * leaves + physical_leaf


def ranges(history, p):
    leaves, width, extra = p
    output = []
    for leaf in range(leaves):
        begin = (leaf * width + min(leaf, extra)) * TILE
        size = (width + int(leaf < extra)) * TILE
        output.append((begin, max(0, min(history - begin, size))))
    return output


def workload(history, groups, revised=False):
    p = plan(history, groups, revised)
    spans = ranges(history, p)
    counts, items = [0] * CORES, [0] * CORES
    for item in range(groups * p[0]):
        group, leaf = decode(item, groups, p[0], revised)
        assert encode(group, leaf, groups, p[0], revised) == item
        valid = spans[leaf][1]
        if valid:
            counts[item % CORES] += (valid + TILE - 1) // TILE
            items[item % CORES] += 1
    return dict(plan=list(p), per_core_steps=counts, max_steps=max(counts),
                min_steps=min(counts), per_core_leaves=items, total_steps=sum(counts))


def main():
    inverse_cases = 0
    for groups in range(1, 97):
        for leaves in range(1, MAX_LEAVES + 1):
            observed = set()
            for item in range(groups * leaves):
                pair = decode(item, groups, leaves, True)
                assert pair not in observed
                observed.add(pair)
                assert encode(*pair, groups, leaves, True) == item
            assert observed == {(g, l) for g in range(groups) for l in range(leaves)}
            inverse_cases += 1
    coverage_cases = 0
    for groups in (1, 4, 8, 16, 23, 24, 25, 28, 32, 48, 64, 96):
        for history in range(0, 262145, 1024):
            for delta in (-1, 0, 1):
                value = min(262144, max(0, history + delta))
                p = plan(value, groups, True)
                active = [(a, n) for a, n in ranges(value, p) if n]
                assert p[0] <= MAX_LEAVES and 1 <= p[1] <= 8
                assert sum(n for _, n in active) == value
                assert not active or active[0][0] == 0
                assert all(a + n == b for (a, n), (b, _) in zip(active, active[1:]))
                coverage_cases += 1
    raw = HERE / "score_pressure_h131072_b28_v1/phase_clock/score_pressure.json"
    diag = json.loads(raw.read_text())
    recorded = [next(row for row in diag["rows"] if row["core"] == core)["ack_count"] for core in range(CORES)]
    reference = workload((131072 + 6) // 4, 28)
    assert reference["per_core_steps"] == recorded
    representatives = {str(groups): {"reference": workload(32769, groups),
                                     "aligned_rotation": workload(32769, groups, True)}
                       for groups in (4, 8, 16, 23, 24, 25, 28, 32, 48)}
    result = dict(passed=True, inverse_cases=inverse_cases, coverage_cases=coverage_cases,
                  reference_matches_recorded_b28_counts=True, diagnostic=str(raw),
                  cases=representatives, device_validated=False, performance_eligible=False)
    (HERE / "LEAF_BALANCE_MODEL.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for groups, case in representatives.items():
        print(groups, "max", case["reference"]["max_steps"], "->", case["aligned_rotation"]["max_steps"],
              "leaves", case["reference"]["plan"][0], "->", case["aligned_rotation"]["plan"][0])


if __name__ == "__main__":
    main()
