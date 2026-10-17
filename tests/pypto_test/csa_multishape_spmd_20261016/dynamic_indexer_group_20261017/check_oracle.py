# SPDX-License-Identifier: Apache-2.0
"""执行正式分组源码控制流，对独立请求枚举和对角矩阵作CPU校验。

不导入编译器/设备。仅移除DSL装饰器/类型，以NumPy执行显式FP16表达式；
这不模拟A3舍入、cache或核间同步。常规叶计划另与canonical完整AST核对。
"""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
CANONICAL = SOURCE.parent / "deepseek_v4_flash_csa/decode_indexer.py"


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


def helpers():
    reads, stores = [], []

    def read(tensor, offsets):
        assert all(0 <= v < size for v, size in zip(offsets, tensor.shape))
        reads.append((id(tensor), tuple(offsets)))
        return tensor[tuple(offsets)].item()

    def write(tensor, offsets, value):
        assert all(0 <= v < size for v, size in zip(offsets, tensor.shape))
        tensor[tuple(offsets)] = value

    def load(tensor, offsets, sizes):
        assert all(v >= 0 and v + size <= cap for v, size, cap in zip(offsets, sizes, tensor.shape))
        reads.extend((id(tensor), (i,)) for i in range(offsets[0], offsets[0] + sizes[0]))
        return tensor[tuple(slice(v, v + size) for v, size in zip(offsets, sizes))].copy()

    def store(tile, offsets, tensor):
        assert all(v >= 0 and v + size <= cap for v, size, cap in zip(offsets, tile.shape, tensor.shape))
        stores.append((id(tensor), tuple(offsets), tile.shape))
        tensor[tuple(slice(v, v + size) for v, size in zip(offsets, tile.shape))] = tile
        return tensor

    def assemble(dest, tile, offsets):
        result = dest.copy()
        store(tile, offsets, result)
        return result

    def cast(value, dtype, **_):
        return value.astype(dtype) if isinstance(value, np.ndarray) else dtype(value).item()

    pl = SimpleNamespace(
        dynamic=lambda name: name,
        INDEX=np.int64,
        INT32=np.int32,
        INT64=np.int64,
        FP16=np.float16,
        FP32=np.float32,
        cast=cast,
        range=range,
        unroll=range,
        min=min,
        max=max,
        mul=np.multiply,
        read=read,
        write=write,
        load=load,
        store=store,
        tensor=SimpleNamespace(read=read, dim=lambda x, axis: x.shape[axis]),
        tile=SimpleNamespace(
            full=lambda shape, dtype, value: np.full(shape, value, dtype=dtype), assemble=assemble, reshape=np.reshape
        ),
    )
    namespace = {"pl": pl, "M": SimpleNamespace(index_n_heads=64, max_position_embeddings=1048576)}
    for name in ("request_geometry.py", "indexer_group_stages.py"):
        tree = ast.fix_missing_locations(Plain().visit(ast.parse((SOURCE / name).read_text())))
        exec(compile(tree, str(SOURCE / name), "exec"), namespace)
    return namespace, reads, stores


def source_math_audit():
    def body(path, function):
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        return ast.Module(body=node.body[1:], type_ignores=[])

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            aliases = {
                "BUFFERED_LONG_SCORE_TILE": "SCORE_TILE",
                "TOPK_CANDIDATES_PER_LEAF": "CANDIDATES_PER_LEAF",
                "TOPK_SCORE_WORKERS": "SCORE_WORKERS",
                "TOPK_MAX_LEAVES": "MAX_LEAVES",
            }
            node.id = aliases.get(node.id, node.id)
            return node

    reference = Rename().visit(body(CANONICAL, "indexer_long_leaf_plan"))
    actual = body(SOURCE / "indexer_group_stages.py", "indexer_long_leaf_plan")
    assert ast.dump(reference) == ast.dump(actual), "叶计划完整数学体发生变化"

    # 逐lane改为单行load，数值链仍使用canonical四个表达式，保留两次cast顺序。
    def assignments(path):
        result = {}
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("query_scale_half", "query_weight_half", "head_coefficients"):
                    result[node.targets[0].id] = ast.dump(node.value)
        return result

    assert assignments(CANONICAL) == assignments(SOURCE / "indexer_group_stages.py")
    return {"leaf_full_mathematical_AST_equal": True, "fp16_cast_multiply_AST_equal": True}


def run():
    ns, reads, stores = helpers()
    rng = np.random.default_rng(20261017)
    cases = [
        ([], []),
        ([2, 2, 2], [128, 256, 64]),
        ([4, 2], [128, 256]),
        ([1, 13, 9], [0, 18, 17]),
        ([3, 2, 6], [127, -1, 257]),
        ([0, 1, 1], [-1, 0, 1]),
        ([6, 6], [-1, -1]),
        ([1, 7, 2, 17], [1048576, 8191, 131071, 7]),
    ]
    cases += [
        ([6] * batch, [history] * batch)
        for history, batch in [
            (131072, 4),
            (131072, 8),
            (131072, 16),
            (131072, 24),
            (131072, 28),
            (8192, 16),
            (8192, 32),
        ]
    ]
    cases += [
        ([int(v) for v in rng.integers(0, 20, size=5)], [int(v) for v in rng.integers(0, 262144, size=5)])
        for _ in range(8)
    ]
    reports = []
    geometry_checks = coefficient_cases = 0
    for case_id, (widths, histories) in enumerate(cases):
        q = np.array([0, *np.cumsum(widths)], np.int32)
        total = int(q[-1])
        lengths = np.array([w + h if h >= 0 else 0 for w, h in zip(widths, histories)], np.int32)
        positions = np.full(total, -987654, np.int64)
        for r, (w, h) in enumerate(zip(widths, histories)):
            if h >= 0:
                positions[q[r] : q[r + 1]] = np.arange(h, h + w)
        if case_id % 3 == 0 and total:
            positions[total // 2] = -321  # 陈旧causal不能污染系数/可见范围。
        scales = rng.uniform(0.00001, 0.2, (total + 5, 64)).astype(np.float32)
        weights = rng.uniform(-8, 8, (total + 7, 64)).astype(np.float32)
        for group_size in (2, 6):
            groups, expected_offsets = [], []
            for r, (width, history) in enumerate(zip(widths, histories)):
                expected_offsets.append(len(groups))
                if history >= 0 and width > 0:
                    token_list = list(range(int(q[r]), int(q[r + 1])))
                    groups.extend((r, token_list[i : i + group_size]) for i in range(0, len(token_list), group_size))
            expected_offsets.append(len(groups))
            descriptor_base = (len(widths) + 16) // 16 * 16
            offsets = np.full(descriptor_base + 4 * total + 3, -42, np.int32)
            plan = np.full(8, -42, np.int32)
            for balanced in (False, True):
                ns["indexer_group_plan_stage"](q, lengths, positions, offsets, plan, total, 1, group_size, balanced, 1)
                if not balanced:
                    assert np.all(offsets == -42) and np.all(plan == -42)
                ns["indexer_group_plan_stage"](q, lengths, positions, offsets, plan, total, 0, group_size, balanced, 1)
                assert offsets[: len(widths) + 1].tolist() == expected_offsets
                assert np.all(offsets[len(widths) + 1 : descriptor_base] == -42)
                assert np.all(offsets[descriptor_base + 4 * len(groups) :] == -42)
                assert plan[0] == len(groups)
                expected_history = max(
                    [min(int(lengths[r]) // 4, 262144) for r, w in enumerate(widths) if w > 0 and lengths[r] > 0] or [0]
                )
                assert plan[1] == expected_history
                leaves, leaf_tiles, extra = (int(plan[i]) for i in (2, 3, 4))
                items = {ns["indexer_work_item"](i, len(groups), leaves) for i in range(len(groups) * leaves)}
                assert items == {(g, leaf) for g in range(len(groups)) for leaf in range(leaves)}
                assert ns["indexer_work_item"](0, 0, leaves) == (-1, -1)
                for group, (r, queries) in enumerate(groups):
                    good = [t for t in queries if positions[t] == histories[r] + t - q[r]]
                    expected_visible = max(
                        [min(int(lengths[r]) // 4, (int(positions[t]) + 1) // 4, 262144) for t in good] or [0]
                    )
                    reads.clear()
                    actual = ns["indexer_decode_group"](q, lengths, positions, offsets, group, group_size)
                    assert all(identity == id(offsets) for identity, _ in reads), "消费者不应重复读取Q/L/position数值"
                    assert actual == (r, queries[0], len(queries), expected_visible)
                    consumed, prev_end = 0, 0
                    for leaf in range(leaves):
                        begin, capacity, valid = ns["indexer_leaf_range"](leaf, leaf_tiles, extra, expected_visible)
                        assert begin == prev_end and capacity > 0
                        consumed += valid
                        prev_end = begin + capacity
                    assert consumed == expected_visible, (case_id, group_size, balanced, expected_visible, consumed)
                    geometry_checks += 1
            expected = np.zeros((16 * len(groups), group_size * 64), np.float16)
            permitted = set()
            for group, (r, queries) in enumerate(groups):
                for lane, t in enumerate(queries):
                    if positions[t] == histories[r] + t - q[r]:
                        permitted.add(t)
                        expected[group * 16 + lane, lane * 64 : (lane + 1) * 64] = scales[t].astype(
                            np.float16
                        ) * weights[t].astype(np.float16)
            for workers in (1, 3, 48):
                guard = np.full((16 * len(groups) + 5, group_size * 64), -29, np.float16)
                output = guard[1:-1]
                reads.clear()
                stores.clear()
                for worker in reversed(range(workers)):
                    ns["indexer_head_coefficients_stage"](
                        scales, weights, positions, q, lengths, offsets, output, worker, workers, group_size
                    )
                assert np.array_equal(output[: len(expected)], expected)
                assert np.all(output[len(expected) :] == -29) and np.all(guard[[0, -1]] == -29)
                gm_stores = [index[0] for identity, index, _ in stores if identity == id(output)]
                assert sorted(gm_stores) == list(range(0, 16 * len(groups), 16)), "组存在重复writer或漏写"
                assert all(index[0] in permitted for identity, index in reads if identity in (id(scales), id(weights)))
                coefficient_cases += 1
            reports.append(
                {"case": case_id, "widths": widths, "group_size": group_size, "groups": len(groups), "passed": True}
            )
    return {
        "passed": True,
        "device_execution": False,
        "source_math": source_math_audit(),
        "geometry_groups_checked": geometry_checks,
        "coefficient_owner_cases": coefficient_cases,
        "cases": reports,
        "limit": "CPU NumPy不是A3 FP16舍入/同步/最终TopK证明",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
