# SPDX-License-Identifier: Apache-2.0
"""B16长历史的16384候选/叶策略，核内保留动态长度和分片均衡。

原每叶8192候选，128K+B16/S6对应32769个压缩候选，均衡后6叶。
扩大叶上限后为3叶，每个worker仍做22个1024分数块，Query/系数加载
由4轮减少到2轮。分数数学和1024块内通信不变；TopK合并分组改变，
不能仅以数学等价断言输出完全一致，仍须真机exact与图重放验证。
"""

import ast
import copy
import json
from pathlib import Path


FUNCTIONS = (
    "indexer_score_topk_native_pair_aic",
    "indexer_score_topk_native_pair_aiv",
    "indexer_topk_query_merge__2",
)


def leaf_plan(count, capacity):
    tiles = (count + 1023) // 1024
    leaves = max((count + capacity - 1) // capacity, 1)
    leaves = (leaves + 2) // 3 * 3
    return leaves, max(tiles // leaves, 1), tiles % leaves


def check_plan():
    # 不仅核对固定131072，覆盖运行时长度边界及不同请求的有效前缀。
    lengths = sorted(set(range(0, 4097)) | set(range(32760, 32782))
                     | {k + d for k in (8192, 16384, 65536, 131072, 262144)
                        for d in (-1, 0, 1) if 0 <= k + d <= 262144})
    for count in lengths:
        leaves, tiles, extra = leaf_plan(count, 16384)
        ranges = [((i * tiles + min(i, extra)) * 1024,
                   (tiles + int(i < extra)) * 1024) for i in range(leaves)]
        if not 1 <= leaves <= 32 or any(size > 16384 for _, size in ranges):
            raise ValueError("叶数超出pair_arena或Cube分数迭代容量")
        for prefix in sorted({0, min(1, count), count // 3, max(count - 1, 0), count}):
            active = [(begin, min(size, prefix - begin)) for begin, size in ranges if begin < prefix]
            offset = 0
            for begin, size in active:
                if begin != offset:
                    raise ValueError("运行时有效候选范围存在重叠或缺口")
                offset += size
            if offset != prefix:
                raise ValueError("有效候选没有完整覆盖")
            # 最终merge实际使用的公式；必须等于生产者发布的有效叶数。
            visible_tiles = (prefix + 1023) // 1024
            large_span = extra * (tiles + 1)
            large_count = (min(visible_tiles, large_span) + tiles) // (tiles + 1)
            small_count = (max(visible_tiles - large_span, 0) + tiles - 1) // tiles
            if large_count + small_count != len(active):
                raise ValueError("最终merge读取未发布的叶或遗漏有效叶")
    workload = {}
    for capacity in (8192, 16384):
        leaves, tiles, extra = leaf_plan(32769, capacity)
        per_core = []
        for worker in range(24):
            calls = list(range(worker, 16 * leaves, 24))
            counts = []
            for item in calls:
                leaf = item // 16
                start = (leaf * tiles + min(leaf, extra)) * 1024
                count = min((tiles + int(leaf < extra)) * 1024, 32769 - start)
                counts.append(max(0, (count + 1023) // 1024))
            per_core.append({"worker": worker, "query_loads": len(calls), "score_blocks": sum(counts)})
        workload[str(capacity)] = {"leaves": leaves, "tiles": tiles, "extra": extra,
                                   "per_core": per_core}
    return {"runtime_lengths_checked": len(lengths), "visible_prefixes_per_length": 5,
            "coverage_and_merge_count_passed": True, "capacity_bound_passed": True,
            "h131072_b16_workload": workload, "precision_and_performance_proven": False}


def use_larger_leaves(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("16384叶候选先限定T96，不套用其他batch策略")
    names = {call["logical_function"] for call in schedule["call_bindings"].values()}
    if not {"indexer_score_topk_native_pair", "indexer_topk_query_merge__2"} <= names:
        raise ValueError("只允许S6长历史的Cube/Vector与均衡TopK合并路径")
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    changes = []
    for name in FUNCTIONS:
        function = functions[name]
        count = [node for node in function.body if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) and node.target.id.startswith("leaf_count_")
                 and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.max"]
        default = [node for node in function.body if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.target.id.startswith("leaf_tiles_")
                   and isinstance(node.value, ast.Constant) and node.value.value == 8]
        if len(count) != 1 or len(default) != 1:
            raise ValueError(f"原叶计划结构改变：{name}")
        division = count[0].value.args[0]
        if (not isinstance(division, ast.BinOp) or not isinstance(division.op, ast.FloorDiv)
                or ast.unparse(division.right) != "8192"
                or not isinstance(division.left, ast.BinOp)
                or not isinstance(division.left.op, ast.Add)
                or ast.unparse(division.left.right) != "8191"):
            raise ValueError(f"原叶数不再按8192向上取整：{name}")
        division.right = ast.Constant(16384)
        division.left.right = ast.Constant(16383)
        default[0].value = ast.Constant(16)
        changes.append({"function": name, "leaf_count_target": count[0].target.id,
                        "default_tile_target": default[0].target.id})
        if name.endswith("_aic"):
            span = [node for node in ast.walk(function) if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name) and node.target.id.startswith("buf_lane_span_")]
            if len(span) != 1 or ast.unparse(span[0].value.args[1]) != "4096":
                raise ValueError("Cube分数迭代截断不是已知的8个1024块")
            span[0].value.args[1] = ast.Constant(8192)
            changes[-1]["aic_max_score_blocks"] = 16
    result = copy.deepcopy(catalog)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_indexer_leaf16384.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# 长历史T96的16384候选/叶实验，长度仍在核内读取。\n"
                    + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    audit = {"old_capacity": 8192, "new_capacity": 16384, "changes": changes,
             "root_abi_unchanged": True, "gm_allocations_unchanged": True,
             "pair_arena_max_roots_per_query": 32, "plan": check_plan()}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result["body_bundle"] = str(path)
    return result, audit
