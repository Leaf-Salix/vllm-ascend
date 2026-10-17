# SPDX-License-Identifier: Apache-2.0
"""四份完整数学体AST对照与少量边界模型；不模拟硬件排序结果，不执行NPU。

只剔除函数说明字符串，函数体的每条原运算/分支/return必须逐AST一致。
容量注解和inline作用域是迁移合同，不在数学体对比中。尾mask模型检查真实
有效列、逻辑index及root层级读取范围；它不能代替设备同分排序精度验证。
"""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_topk_stages.py"
NAMES = ["merge2_top512_pairs", "merge_top512_roots", "indexer_topk_half_leaf", "indexer_topk_segment_pairs"]


def functions(path):
    return {f.name: f for f in ast.parse(path.read_text()).body if isinstance(f, ast.FunctionDef)}


def mathematical_body(function):
    body = function.body
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def math_audit():
    old = functions(HERE / "reference_decode_indexer.py")
    new = functions(SOURCE)
    comparisons = {name: mathematical_body(old[name]) == mathematical_body(new[name]) for name in NAMES}
    assert all(comparisons.values())
    source = ast.parse(SOURCE.read_text())
    calls = [ast.unparse(node.func) for node in ast.walk(source) if isinstance(node, ast.Call)]
    assert not any("spmd" in call or "submit" in call or "notify" in call or "sync_" in call for call in calls)
    symbols = {node.id for node in ast.walk(source) if isinstance(node, ast.Name)}
    assert not {"B", "S", "T", "T_PAD", "SCORE_ARENA_ROWS", "TOPK_ARENA_ROWS"} & symbols
    return comparisons


def tail_model(count, segment=False, offset=0, logical_begin=8177):
    widths = [512, 1024, 2048] if segment else [512, 1024, 2048, 2560, 3072, 4096]
    assert 1 <= count <= widths[-1]
    width = next(w for w in widths if count <= w)
    assert offset >= 0 and offset + width <= 8192
    # 原load的valid_shape仅允许[0,count)被当作已写score。fillpad覆盖其余列。
    live = set(range(count))
    masked = set(range(count, width))
    assert not live & masked and live | masked == set(range(width))
    indices = [logical_begin + lane for lane in range(width)]
    assert indices[:count] == list(range(logical_begin, logical_begin + count))
    assert all(0 <= index < 2**31 for index in indices)
    return {"count": count, "load_width": width, "score_offset": offset, "padding_columns": width - count}


def merge_model(count, arena_base=5, capacity_margin=3):
    assert count >= 1
    reads = [arena_base]
    rounds = []
    # root内部的优先序仅用于检查原merge输入序列；不宣称硬件mrgsort同分行为已实测。
    priority = [0]
    for child in range(1, count, 3):
        incoming = list(range(child, min(child + 3, count)))
        reads.extend(arena_base + child_index for child_index in incoming)
        priority = list(reversed(incoming)) + priority
        rounds.append({"child": child, "incoming": list(reversed(incoming)), "fan_in": len(incoming) + 1})
    capacity = arena_base + count + capacity_margin
    assert reads == list(range(arena_base, arena_base + count))
    assert all(0 <= row < capacity for row in reads)
    assert sorted(priority) == list(range(count))
    assert priority == list(reversed(range(count)))
    assert all(2 <= row["fan_in"] <= 4 for row in rounds)
    return {"count": count, "arena_base": arena_base, "pair_capacity": capacity, "rounds": rounds}


def main():
    comparisons = math_audit()
    boundaries = [
        1,
        31,
        32,
        33,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        2559,
        2560,
        2561,
        3071,
        3072,
        3073,
        4095,
        4096,
    ]
    half = [tail_model(count) for count in boundaries]
    segment = [
        tail_model(count, segment=True, offset=offset) for count in boundaries if count <= 2048 for offset in (0, 2048)
    ]
    roots = [merge_model(count) for count in (1, 2, 3, 4, 5, 7, 8, 31, 32, 64)]
    report = {
        "passed": True,
        "scope": "四个完整数学体AST+少量候选尾mask/根计数模型",
        "source": str(SOURCE),
        "reference_snapshot": str(HERE / "reference_decode_indexer.py"),
        "full_mathematical_AST_equal": comparisons,
        "annotation_changes": "score/pair行容量独立pl.dynamic；写pair的两个helper标InOut。数学体未改。",
        "half_leaf_cases": half,
        "segment_cases": segment,
        "root_count_cases": roots,
        "zero_count_policy": "0候选或0根不在本helper合同；外层须写哨兵或跳过，不以容量代替有效数。",
        "model_limit": "纯模型只检查索引/覆盖与源级merge输入序，未模拟sort32/mrgsort硬件，未测试设备同分结果。",
        "device_execution": False,
        "performance_claim": False,
    }
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "full_math_bodies": len(comparisons),
                "half_leaf": len(half),
                "segment": len(segment),
                "root_counts": len(roots),
            }
        )
    )


if __name__ == "__main__":
    main()
