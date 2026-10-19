# SPDX-License-Identifier: Apache-2.0
"""在同一叶分配中消除新增的动态除模，保持本轮计划覆盖/owner逐项相同。"""
import ast
import json
from pathlib import Path

from check_leaf_balance_model import CORES, DIVISORS, MAX_LEAVES, decode, encode, plan

HERE = Path(__file__).resolve().parent


def once(source, old, new):
    assert source.count(old) == 1, old[:90]
    return source.replace(old, new)


def main():
    # 证明本版本planner的可达范围，而非把任意外部plan当作受支持输入。
    plan_cases = 0
    for groups in range(1, 97):
        for boundary in range(0, 262145, 1024):
            for delta in (-1, 0, 1):
                history = min(262144, max(0, boundary + delta))
                leaves, _, _ = plan(history, groups, True)
                if groups >= CORES and leaves <= CORES:
                    assert CORES % leaves == 0
                plan_cases += 1
    mapping_cases = 0
    for groups in range(CORES, 97):
        for leaves in DIVISORS:
            for group in range(groups):
                old_rotation = (group // (CORES // leaves)) % leaves
                rotation = (group % CORES) * leaves // CORES
                assert rotation == old_rotation
                for physical in range(leaves):
                    logical = physical + rotation
                    if logical >= leaves:
                        logical -= leaves
                    inverse = logical - rotation
                    if inverse < 0:
                        inverse += leaves
                    item = group * leaves + physical
                    assert decode(item, groups, leaves, True) == (group, logical)
                    assert encode(group, logical, groups, leaves, True) == group * leaves + inverse
                    mapping_cases += 1
    source_dir = HERE / "source_leaf_balanced"
    target = HERE / "source_leaf_balanced_fast"
    target.mkdir(exist_ok=False)
    for path in source_dir.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    path = target / "indexer_group_stages.py"
    source = path.read_text()
    source = once(source, """        if groups >= SCORE_WORKERS and leaves > 0 and leaves <= SCORE_WORKERS:
            if SCORE_WORKERS % leaves == 0:
                groups_per_round = SCORE_WORKERS // leaves
                rotation = (group // groups_per_round) % leaves
""", """        if groups >= SCORE_WORKERS and leaves > 0 and leaves <= SCORE_WORKERS:
            # 同版本planner保证该域的leaves是24的约数；小历史为1，大于24
            # 则不旋转。不接收外部任意旧plan。恒等式把三个动态除模改为
            # 常量24的除模：floor(group/(24/leaves)) % leaves
            # == floor((group % 24)*leaves/24)，乘积至多23*24，不会溢出。
            rotation = (group % SCORE_WORKERS) * leaves // SCORE_WORKERS
""")
    source = once(source, """                if rotation != 0:
                    leaf = (leaf + rotation) % leaves
""", """                if rotation != 0:
                    # 0<=leaf,rotation<leaves，所以至多减一次即可回到范围内。
                    leaf = leaf + rotation
                    if leaf >= leaves:
                        leaf = leaf - leaves
""")
    ast.parse(source)
    path.write_text(source)
    path = target / "attention_dependencies.py"
    source = path.read_text()
    source = once(source, """                if rotation != 0:
                    physical_leaf = (leaf + leaves - rotation) % leaves
""", """                if rotation != 0:
                    # leaf与rotation都小于leaves，反解只需一次条件加法。
                    physical_leaf = leaf - rotation
                    if physical_leaf < 0:
                        physical_leaf = physical_leaf + leaves
""")
    ast.parse(source)
    path.write_text(source)
    changed = sorted(p.name for p in source_dir.glob("*.py") if p.read_bytes() != (target / p.name).read_bytes())
    assert changed == ["attention_dependencies.py", "indexer_group_stages.py"]
    report = dict(passed=True, reachable_plan_cases=plan_cases, mapping_cases=mapping_cases,
                  changed=changed, same_logical_work_and_owners=True,
                  formal_source_changed=False, device_validated=False,
                  note="仅可达计划的整数等价证明；不代替真机同步/完整收益")
    (HERE / "LEAF_BALANCE_FAST_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
