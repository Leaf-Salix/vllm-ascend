"""无需导入NPU/PyPTO的候选数学及半叶有效范围审计。"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def check_math() -> dict:
    tree = ast.parse((PACKAGE / "indexer_topk_stages.py").read_text())
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    original = funcs["indexer_topk_half_leaf"]
    stage_tree = ast.parse((PACKAGE / "indexer_score_stages.py").read_text())
    candidate_branch = next(
        node
        for node in ast.walk(stage_tree)
        if isinstance(node, ast.If)
        and "LOCAL_HALF_SCORE_TILE" in ast.unparse(node.test)
        and any(
            isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == "ub_half_logical_begin_i32"
            for statement in node.body
        )
    )
    candidate = copy.deepcopy(original)
    candidate.body = [original.body[0]] + candidate_branch.body[1:]

    class NormalizeNames(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = node.id.removeprefix("ub_half_")
            node.id = {
                "buf_half_data": "score_data",
                "buf_half_begin": "logical_begin",
                "buf_half_valid": "valid_count",
                "buf_half_slot": "output_slot",
            }.get(node.id, node.id)
            return node

    candidate = NormalizeNames().visit(candidate)

    class NormalizeAccess(ast.NodeTransformer):
        def visit_Call(self, node):
            name = ast.unparse(node.func)
            if name == "pl.load" and ast.unparse(node.args[0]) == "score_arena":
                shape = node.args[2]
                valid = next(x.value for x in node.keywords if x.arg == "valid_shape")
                return ast.Call(func=ast.Name(id="half_leaf_data", ctx=ast.Load()), args=[shape, valid], keywords=[])
            if name == "pl.set_validshape" and isinstance(node.args[0], ast.Call):
                source = node.args[0]
                if ast.unparse(source.func) == "pl.tile.extract" and ast.unparse(source.args[0]) == "score_data":
                    assert ast.unparse(source.args[1]) == "0" and ast.unparse(source.args[2]) == "0"
                    shape = source.args[3]
                    valid = ast.List(elts=node.args[1:], ctx=ast.Load())
                    return ast.Call(
                        func=ast.Name(id="half_leaf_data", ctx=ast.Load()), args=[shape, valid], keywords=[]
                    )
            return self.generic_visit(node)

    def body(func):
        nodes = copy.deepcopy(func.body[1:])
        return ast.dump(NormalizeAccess().visit(ast.Module(body=nodes, type_ignores=[])), include_attributes=False)

    assert body(original) == body(candidate), "TopK数学/分支/输入次序不一致"
    calls = [
        ast.unparse(n)
        for n in ast.walk(candidate)
        if isinstance(n, ast.Call) and ast.unparse(n.func) in {"pl.sort32", "pl.mrgsort", "pl.tile.mrgsort"}
    ]
    return dict(normalized_body_equal=True, sort_merge_calls=len(calls), branches=6)


def check_geometry() -> dict:
    step = 384
    capacity = 4096
    buffer_cols = (capacity + step - 1) // step * step
    cases = 0
    # 叶全部可能的非空长度；每组query可比组上界早若干候选，含失活query。
    # 同时覆盖768不整除8192、尾组和短历史两个AIV的不同有效区间。
    for group_valid in range(1, 8193):
        tile_count = (group_valid + 767) // 768
        lane_span = min(tile_count * step, capacity)
        iterations = (lane_span + step - 1) // step
        assert iterations * step <= buffer_cols
        for lane in (0, 1):
            for query_visible in {0, group_valid, max(group_valid - 1, 0), max(group_valid - 2, 0), group_valid // 2}:
                half_valid = max(min(query_visible - lane * lane_span, lane_span), 0)
                old_prefix = 0
                for iteration in range(iterations):
                    begin = iteration * step
                    valid = max(min(query_visible - begin - lane * lane_span, step), 0)
                    if valid:
                        assert begin <= old_prefix
                        old_prefix = max(old_prefix, begin + valid)
                # 原GM可以多写末尾128列，但最终只按half_valid排序；UB同样覆盖此前缀。
                assert half_valid <= old_prefix
                assert half_valid <= iterations * step
                assert half_valid <= capacity
                cases += 1
    return dict(cases=cases, ub_bytes=2 * buffer_cols * 4, buffer_cols=buffer_cols, passed=True)


def main():
    result = dict(
        math=check_math(), geometry=check_geometry(), device_precision_validated=False, performance_measured=False
    )
    (HERE / "source_geometry_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
