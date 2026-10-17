# SPDX-License-Identifier: Apache-2.0
"""发布阶段的源码数学、根计数、真实Q/L地址和worker模型；不执行A3排序。

同源AST解释实际Scalar分支。排序/merge用明确标记的payload桩，只检查读取根
范围、原位根0所有权、输出填空和index尾掩码；不把桩当作数值排序oracle。
排序平局顺序通过canonical的完整leaf/merge调用AST直接核对。
"""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
SOURCE = PACKAGE / "indexer_publish_stages.py"
CANONICAL = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_indexer.py"
NEG = np.float32(-3.4028234663852886e38)


def funcs(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


class Names(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(id={"multiway": "BALANCED", "half_count": "root_count"}.get(node.id, node.id), ctx=node.ctx), node
        )


def source_audit():
    old, new = funcs(CANONICAL), funcs(SOURCE)
    assert [dump(n) for n in old["indexer_topk_leaf_publish"].body[1:]] == [
        dump(n) for n in new["indexer_topk_leaf_publish"].body[1:]
    ]

    def merge_branch(function):
        return next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If)
            and n.body
            and isinstance(n.body[0], ast.Assign)
            and ast.unparse(n.body[0].targets[0]) == "root_pairs"
        )

    assert dump(Names().visit(copy.deepcopy(merge_branch(old["indexer_topk_query_merge_one"])))) == dump(
        merge_branch(new["indexer_pair_publish_stage"])
    )

    def publish_tail(function):
        scope = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If)
            and any(isinstance(c, ast.Assign) and ast.unparse(c.targets[0]) == "root_scores" for c in n.body)
        )
        start = next(
            i
            for i, n in enumerate(scope.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "root_scores"
        )
        return scope.body[start:]

    assert [dump(n) for n in publish_tail(old["indexer_topk_query_merge_one"])] == [
        dump(n) for n in publish_tail(new["indexer_pair_publish_stage"])
    ]
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n, ast.Call)]
    assert not any(c in ("pl.spmd", "pl.at", "pl.tile.get_block_idx", "pl.system.submit") for c in calls)
    forbidden = [
        n
        for n in ast.walk(ast.parse(SOURCE.read_text()))
        if isinstance(n, ast.Name) and n.id in ("S", "DECODE_SEQ", "T_PAD", "LONG_S6_MIN_QUERY_ROWS")
    ]
    assert not forbidden
    return {
        "leaf_sort_full_math_ast_exact": True,
        "pair_merge_tree_ast_exact_after_variable_rename": True,
        "scores_indices_publish_tail_ast_exact": True,
        "no_fixed_S_T_pad_or_T_based_strategy_guess": True,
        "no_nested_dispatch": True,
        "numerical_sort_not_executed_on_cpu": True,
    }


class Plain(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        return self.generic_visit(node)


class Interpreter:
    def __init__(self):
        self.events = []
        self.pl = SimpleNamespace(
            dynamic=lambda name: name,
            INDEX="INDEX",
            INT32="INT32",
            FP32="FP32",
            range=range,
            min=min,
            max=max,
            cast=lambda x, dtype: int(x),
            read=self.read,
            write=self.write,
            load=self.load,
            store=self.store,
            tensor=SimpleNamespace(dim=lambda x, i: x.shape[i], read=self.read),
        )
        self.pl.tile = SimpleNamespace(
            full=lambda shape, dtype, value: np.full(shape, value, np.int32 if dtype == "INT32" else np.float32),
            gather_mask=lambda p, mask_pattern, output_dtype: p[:, ::2].copy()
            if mask_pattern == "score"
            else p[:, 1::2].copy().view(np.int32),
            MaskPattern=SimpleNamespace(P0101="score", P1010="index"),
            read=self.read,
            write=self.write,
        )
        self.env = {
            "pl": self.pl,
            "M": SimpleNamespace(index_n_heads=64, max_position_embeddings=1048576),
            "FP32_NEG_INF": float(NEG),
            "IDX_TOPK": 512,
            "TOPK_PAIR_WIDTH": 1024,
            "PAIR_ROWS_DYN": "PAIR",
            "SCORE_ROWS_DYN": "SCORE",
        }
        for name in ("request_geometry.py", "indexer_group_stages.py", "indexer_publish_stages.py"):
            exec(
                compile(
                    ast.fix_missing_locations(Plain().visit(ast.parse((PACKAGE / name).read_text()))), name, "exec"
                ),
                self.env,
            )
        self.env["merge2_top512_pairs"] = self.merge2
        self.env["merge_top512_roots"] = self.merge_many
        self.env["indexer_topk_leaf_publish"] = self.leaf_stub

    def read(self, array, index):
        assert all(0 <= i < n for i, n in zip(index, array.shape))
        return array[tuple(index)].item()

    def write(self, array, index, value):
        assert all(0 <= i < n for i, n in zip(index, array.shape))
        array[tuple(index)] = value

    def load(self, array, origin, shape):
        assert all(0 <= i and i + s <= n for i, s, n in zip(origin, shape, array.shape))
        self.events.append(("load", origin[0], shape[0]))
        return array[origin[0] : origin[0] + shape[0], origin[1] : origin[1] + shape[1]].copy()

    def store(self, value, origin, array):
        assert all(0 <= i and i + s <= n for i, s, n in zip(origin, value.shape, array.shape))
        array[origin[0] : origin[0] + value.shape[0], origin[1] : origin[1] + value.shape[1]] = value

    def merge2(self, array, left, right, out):
        assert left == out and right > left and left // 64 == right // 64
        self.events.append(("merge2", left, right, out))
        array[out] = array[right]

    def merge_many(self, array, base, count):
        assert 0 < count <= 64 and base + count <= len(array)
        self.events.append(("merge_many", base, count))
        return array[base : base + 1].copy()

    def leaf_stub(self, array, query, valid, scores, indices):
        assert 0 < valid <= 8192 and 0 <= query < len(array)
        self.events.append(("leaf", query, valid))
        scores[query] = NEG
        indices[query] = -1
        scores[query, : min(valid, 512)] = 123
        indices[query, : min(valid, 512)] = np.arange(min(valid, 512))


def independent_visible(bounds, lengths, positions, query):
    matches = []
    for r, length in enumerate(lengths):
        if r + 1 < len(bounds):
            begin, end = map(int, bounds[r : r + 2])
            if length > 0 and 0 <= begin < end <= len(positions) and end - begin <= length and begin <= query < end:
                matches.append((int(length), begin, end))
    if len(matches) != 1:
        return 0
    length, begin, end = matches[0]
    if positions[query] != length - (end - begin) + query - begin:
        return 0
    return max(0, min(length // 4, (int(positions[query]) + 1) // 4, 262144))


def root_oracle(visible, plan, roots, balanced):
    if roots not in (1, 2) or int(plan[7]) != roots or int(plan[6]) != balanced:
        return 0
    groups, maxhist, leaves, tiles, extra, group_size, _, _ = map(int, plan)
    if not (0 < visible <= maxhist <= 262144 and 0 < leaves <= 32 and tiles > 0 and 0 <= extra < leaves):
        return 0
    if not balanced and (tiles != 8 or extra != 0):
        return 0
    # 独立按每叶容量铺开，再数visible实际触达几叶，不复用被测反算公式。
    cursor = 0
    count = 0
    for leaf in range(leaves):
        if cursor < visible:
            count += 1
        cursor += (tiles + (leaf < extra)) * 1024
    if cursor < visible:
        return 0
    return count * roots


def case(interpreter, widths, histories, group_size, roots, balanced, workers, mutation=None):
    bounds = np.array([1, *[1 + x for x in np.cumsum(widths)]], np.int32)
    tokens = int(bounds[-1]) + 2
    lengths = np.array([w + h if h >= 0 else 0 for w, h in zip(widths, histories)], np.int32)
    positions = np.full(tokens, 2**31 + 3, np.int64)
    for r, (width, history) in enumerate(zip(widths, histories)):
        if history >= 0:
            positions[bounds[r] : bounds[r + 1]] = np.arange(history, history + width)
    if mutation == "stale" and widths:
        positions[int(bounds[0])] += 4
    offsets = np.full((len(widths) + 16) // 16 * 16 + 4 * tokens, -999, np.int32)
    plan = np.full(8, -999, np.int32)
    interpreter.env["indexer_group_plan_stage"](bounds, lengths, positions, offsets, plan, tokens, 0, group_size, balanced, roots)
    if mutation == "bad_plan_roots":
        plan[7] = 3
    if mutation == "bad_leaf_tiles":
        plan[3] = 0
    if mutation == "bad_extra":
        plan[4] = plan[2]
    pair = np.zeros((tokens * 64 + 3, 1024), np.float32)
    pair[:, ::2] = 123
    pair[:, 1::2] = np.arange(512, dtype=np.int32).view(np.float32)
    if mutation == "short_pair":
        pair = pair[:0]
    score = np.full((tokens + 1, 8192), -777, np.float32)
    score_out = np.full((tokens + 2, 512), -777, np.float32)
    index_out = np.full((tokens + 2, 512), -777, np.int32)
    scores, indices = score_out[1:-1], index_out[1:-1]
    interpreter.events = []
    for worker in reversed(range(workers)):
        if roots:
            interpreter.env["indexer_pair_publish_stage"](
                bounds, lengths, positions, plan, pair, scores, indices, worker, workers, roots, balanced
            )
        else:
            interpreter.env["indexer_single_leaf_publish_stage"](
                bounds, lengths, positions, plan, score, scores, indices, worker, workers
            )
    expected_events = []
    for worker in reversed(range(workers)):
        for query in range(worker, tokens, workers):
            visible = independent_visible(bounds, lengths, positions, query)
            count = root_oracle(visible, plan, roots, balanced) if roots else 0
            valid = (
                (count > 0 and query * 64 + count <= len(pair))
                if roots
                else (0 < visible <= plan[1] <= 8192 and plan[2] == 1 and plan[6] == 0 and plan[7] == 0)
            )
            if valid:
                if roots:
                    if balanced:
                        expected_events.append(("merge_many", query * 64, count))
                    else:
                        expected_events.extend(
                            ("merge2", query * 64, query * 64 + child, query * 64) for child in range(1, count)
                        )
                        expected_events.append(("load", query * 64, 1))
                else:
                    expected_events.append(("leaf", query, visible))
                assert np.array_equal(indices[query, : min(visible, 512)], np.arange(min(visible, 512)))
                assert np.all(indices[query, min(visible, 512) :] == -1)
            else:
                assert np.all(scores[query] == NEG) and np.all(indices[query] == -1)
    assert interpreter.events == expected_events
    assert np.all(score_out[[0, -1]] == -777) and np.all(index_out[[0, -1]] == -777)
    return {
        "tokens": tokens,
        "widths": widths,
        "histories": histories,
        "group_size": group_size,
        "roots": roots,
        "balanced": balanced,
        "workers": workers,
        "mutation": mutation,
        "plan": plan.tolist(),
        "event_count": len(expected_events),
        "passed": True,
    }


def main():
    audit = source_audit()
    interpreter = Interpreter()
    root_cases = 0
    for groups, maxhist in ((4, 32769), (8, 32768), (16, 32769), (3, 8193), (24, 262144), (0, 0)):
        for balanced in (False, True):
            leaves, tiles, extra = (
                interpreter.env["indexer_long_leaf_plan"](maxhist, groups)
                if balanced
                else (max(1, (maxhist + 8191) // 8192), 8, 0)
            )
            for roots in (1, 2):
                plan = np.array([groups, maxhist, leaves, tiles, extra, 2, balanced, roots], np.int32)
                boundaries = {0, 1, 511, 512, 1023, 1024, maxhist}
                cursor = 0
                for leaf in range(leaves):
                    cursor += (tiles + (leaf < extra)) * 1024
                    boundaries.update((cursor - 1, cursor, cursor + 1))
                for visible in sorted(x for x in boundaries if 0 <= x <= maxhist):
                    actual = interpreter.env["indexer_visible_root_count"](visible, plan, roots, balanced)
                    assert actual == root_oracle(visible, plan, roots, balanced)
                    root_cases += 1
    scenarios = [
        ([1, 3, 5], [0, 7, 131071], 2, 2, True),
        ([6] * 4, [131072] * 4, 6, 1, True),
        ([4] * 8, [131071] * 8, 2, 2, True),
        ([6, 3], [4092, 10], 2, 2, False),
        ([1, 3, 2], [3, -1, 7], 2, 2, False),
        ([], [], 2, 2, True),
        ([1] * 8, [4 * x - 1 for x in (1, 511, 512, 2048, 2049, 4096, 4097, 8192)], 1, 0, False),
    ]
    cases = [case(interpreter, *scenario, workers) for scenario in scenarios for workers in (1, 3, 48)]
    for mutation in ("stale", "bad_plan_roots", "bad_leaf_tiles", "bad_extra", "short_pair"):
        cases.append(case(interpreter, [5, 3], [131071, 7], 2, 2, True, 48, mutation))
    # 真T=0必须无query执行；使用空positions/输出，不以leading/trailing dummy替代。
    empty = np.empty(0, np.int32)
    bounds = np.array([0], np.int32)
    pos = np.empty(0, np.int64)
    interpreter.env["indexer_pair_publish_stage"](
        bounds,
        empty,
        pos,
        np.array([0, 0, 1, 8, 0, 2, 0, 2], np.int32),
        np.empty((0, 1024), np.float32),
        np.empty((0, 512), np.float32),
        np.empty((0, 512), np.int32),
        0,
        48,
        2,
        False,
    )
    report = {
        "passed": True,
        "source_audit": audit,
        "root_boundary_cases": root_cases,
        "ownership_geometry_cases": len(cases),
        "cases": cases,
        "true_T0_no_query_work": True,
        "canonical_root_stride": 64,
        "source_AST_scalar_control_interpreted": True,
        "sort_and_merge_payloads_mocked": True,
        "device_execution": False,
        "precision_proven": False,
        "limits": [
            "排序/merge数值仅核对调用AST和参数顺序，CPU桩不证明A3平局结果",
            "不模拟跨核等待、缓存一致性或score生产者真实写入",
            "容量与策略错误的防越界分支不能代替完整入口拒绝无效配置",
        ],
    }
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"passed": True, "root_boundary_cases": root_cases, "geometry_cases": len(cases), "device": False})
    )


if __name__ == "__main__":
    main()
