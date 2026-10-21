# SPDX-License-Identifier: Apache-2.0
"""Score归属置换的Host复核模型；只用于测试，不进入算子或计时路径。"""

import json
from pathlib import Path

GROUPS = 24


def pad16(n):
    return (n + 15) // 16 * 16


def base(requests, tokens):
    return pad16(pad16(requests + 1) + 4 * tokens) + 48 + 4 * tokens + 2 * requests


def expected(plan, visible, requests, tokens):
    groups, _, leaves, leaf_tiles, extra, query_group, balanced, _ = plan
    costs, prefix = [0] * 24, [0] * 24
    forward, inverse = list(range(24)), list(range(24))
    if not (24 <= groups <= 48 and 0 < leaves <= 32 and leaf_tiles > 0
            and 0 <= extra < leaves and 128 < tokens <= 256 and query_group == 6 and balanced == 1):
        return costs + prefix + forward + inverse
    for group in range(groups):
        for leaf in range(leaves):
            begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
            capacity = (leaf_tiles + int(leaf < extra)) * 1024
            valid = max(min(visible[group] - begin, capacity), 0)
            costs[(group * leaves + leaf) % 24] += (valid + 1023) // 1024
    if max(costs) - min(costs) < 2:
        return costs + prefix + forward + inverse
    # 独立枚举原KV的M64与M16循环，而不是照抄设备端的闭式块数公式。
    full = tokens // 64 * 64
    padded = (tokens + 15) // 16 * 16
    row_groups = min(3, max(1, tokens // 64))
    for physical in range(24):
        rows = 0
        for logical in range((physical + 20) % 24, 4 * row_groups, 24):
            group = logical % row_groups
            rows += 64 * len(range(group * 64, full, row_groups * 64))
            rows += 16 * len(range(full + group * 16, padded, row_groups * 16))
        qb_columns = len(range(physical % 20, 128, 20)) if physical >= 4 else 0
        prefix[physical] = rows * 129 + qb_columns
    physical_order = sorted(range(24), key=lambda i: (-prefix[i], i))
    logical_order = sorted(range(24), key=lambda i: (costs[i], i))
    for physical, logical in zip(physical_order, logical_order, strict=True):
        forward[physical], inverse[logical] = logical, physical
    return costs + prefix + forward + inverse


def fixtures():
    cases = []
    for name, groups, tokens, visible in (
        ("b28_uniform", 28, 168, [32769] * 28),
        ("b24_uniform_identity", 24, 144, [32769] * 24),
        ("b28_mixed", 28, 168, [32769 if i % 3 else 8193 for i in range(28)]),
        ("b40_uniform", 40, 240, [32769] * 40),
        ("b28_partial_leaf", 28, 168, [34817] * 28),
        ("b28_large_history", 28, 168, [65537] * 28),
        ("b28_empty_visible", 28, 168, [0] * 28),
        ("inactive_identity", 0, 168, []),
        ("group_underflow_identity", 23, 168, [32769] * 23),
        ("group_overflow_identity", 49, 168, [32769] * 49),
        ("lower_token_identity", 28, 128, [32769] * 28),
        ("upper_token_identity", 28, 257, [32769] * 28),
    ):
        requests = max(groups, 1)
        history = max(visible, default=0)
        leaves = max(1, (history + 8191) // 8192)
        plan = [groups, history, leaves, 8, 0, 6, 1, 1]
        cases.append(dict(name=name, requests=requests, tokens=tokens, visible=visible, plan=plan))
    return cases


def main():
    here = Path(__file__).resolve().parent
    diagnostic = json.loads((here.parent / "csa_critical_path_20261020/CURRENT_SCORE_PRESSURE_ANALYSIS.json").read_text())["28"]
    case = fixtures()[0]
    words = expected(case["plan"], case["visible"], case["requests"], case["tokens"])
    observed = [row["blocks"] for row in diagnostic["score_groups"]]
    assert words[:24] == observed, (words[:24], observed)
    forward, inverse = words[48:72], words[72:96]
    assert sorted(forward) == list(range(24)) and all(inverse[forward[i]] == i for i in range(24))
    report = {
        "scope": "当前结构的工作量模型；旧带探针数据只核对步数，不作为当前性能",
        "observed_blocks_match": True, "diagnostic_task": diagnostic["task"],
        "score_blocks_before": words[:24], "prefix_rank": words[24:48],
        "physical_to_logical": forward, "logical_to_physical": inverse,
        "score_blocks_after": [words[i] for i in forward],
        "total_score_blocks_unchanged": sum(words[:24]) == sum(words[i] for i in forward),
        "estimated_runtime_us": None, "device_candidate_measured": False,
    }
    (here / "SCORE_OWNER_WORK_MODEL.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
