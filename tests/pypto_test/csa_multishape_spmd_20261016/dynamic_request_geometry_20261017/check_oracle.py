# SPDX-License-Identifier: Apache-2.0
"""直接解释正式 helper 的 Python AST，并与独立区间/逐位置 oracle 对照。"""

import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace


def source_path():
    repo = Path(__file__).resolve().parents[4]
    return repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/request_geometry.py"


def interpreted_helpers(path):
    # 不另写一份被测实现；只剥离 DSL 装饰器/类型注解，控制流与表达式来自正式文件。
    class PlainPython(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_FunctionDef(self, node):
            node.decorator_list = []
            node.returns = None
            for argument in node.args.args:
                argument.annotation = None
            return self.generic_visit(node)

    reads = []

    def read(tensor, indices):
        index = indices[0]
        if index < 0 or index >= len(tensor):
            raise AssertionError(f"被测源码越界读取：{index}/{len(tensor)}")
        reads.append((id(tensor), index))
        return tensor[index]

    language = SimpleNamespace(
        dynamic=lambda name: name,
        cast=lambda value, dtype: int(value),
        INDEX="INDEX",
        INT32="INT32",
        range=range,
        tensor=SimpleNamespace(read=read, dim=lambda value, axis: len(value)),
    )
    namespace = {"pl": language}
    tree = ast.fix_missing_locations(PlainPython().visit(ast.parse(path.read_text())))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, reads


def oracle_intervals(bounds, lengths, capacity):
    intervals = {}
    for request, length in enumerate(lengths):
        if length <= 0 or request + 1 >= len(bounds):
            continue
        begin, end = bounds[request : request + 2]
        if 0 <= begin < end <= capacity and end - begin <= length:
            intervals[request] = begin, end
    return intervals


def cases():
    result = []
    for width in (1, 2, 5, 6, 7, 9):
        result.append((f"uniform_s{width}", [0, width, 2 * width, 3 * width],
                       [100 + width, 102 + width, 104 + width], 3 * width))
    result.extend([
        ("mixed_222", [0, 2, 4, 6], [130, 258, 66], 6),
        ("mixed_42", [0, 4, 6], [132, 258], 6),
        ("mixed_33", [0, 3, 6], [131, 259], 6),
        ("mixed_odd", [0, 1, 4, 6, 11], [1, 6, 9, 15], 11),
        ("no_compression", [0, 1, 2], [1, 2], 2),
        ("more_than_two_boundaries", [0, 13, 22], [31, 26], 22),
        ("uniform_dummy_tail", [0, 6, 12, 18], [134, 0, 0], 18),
        ("mixed_dummy_tail", [0, 3, 5, 8], [131, 258, 0], 8),
        ("unused_negative_bounds", [0, 3, -1, -1], [131, 0, 0], 8),
        ("inactive_gap", [0, 2, 4, 7], [6, 0, 11], 7),
        ("all_dummy", [-1, -1, -1], [0, 0], 12),
        ("zero_requests", [0], [], 0),
        ("short_bounds", [0, 2], [6, 9, 11], 8),
        ("negative_start", [-1, 3], [7], 4),
        ("reversed_bounds", [4, 2], [6], 8),
        ("outside_capacity", [0, 9], [20], 8),
        ("query_exceeds_length", [0, 6], [5], 6),
        ("empty_request", [0, 0, 3], [2, 7], 3),
        ("negative_length", [0, 3], [-1], 3),
        ("overlap_rejected", [0, 4, 2, 6], [8, 8, 8], 6),
        ("negative_capacity", [0, 1], [5], -1),
    ])
    rng = random.Random(20261017)
    for sample in range(24):
        widths = [rng.randrange(1, 14) for _ in range(rng.randrange(1, 7))]
        bounds = [0]
        for width in widths:
            bounds.append(bounds[-1] + width)
        lengths = [rng.randrange(0, 1024) + width for width in widths]
        result.append((f"seeded_mixed_{sample}", bounds, lengths, bounds[-1]))
    return result


def main():
    source = source_path()
    helpers, reads = interpreted_helpers(source)
    results = []
    for name, bounds, lengths, capacity in cases():
        intervals = oracle_intervals(bounds, lengths, capacity)
        checked = 0
        for request in range(-1, len(lengths) + 2):
            expected_bounds = (*intervals[request], 1) if request in intervals else (0, 0, 0)
            observed = helpers["request_bounds"](bounds, lengths, request, capacity)
            assert observed == expected_bounds, (name, request, observed, expected_bounds)
            expected = (0, 0, 0)
            if request in intervals:
                # 独立 oracle 逐个绝对位置枚举关闭组，而非重用被测 floor 差公式。
                closed = {}
                for prior, (begin, end) in intervals.items():
                    start = lengths[prior] - (end - begin)
                    closed[prior] = [pos for pos in range(start, lengths[prior]) if (pos + 1) % 4 == 0]
                prefix = sum(len(rows) for prior, rows in closed.items() if prior < request)
                begin, end = intervals[request]
                history_start = lengths[request] - (end - begin)
                complete_before = len(range(3, history_start, 4))
                expected = (len(closed[request]), prefix - complete_before - 1, 1)
            observed = helpers["compression_geometry"](bounds, lengths, request, capacity)
            assert observed == expected, (name, request, observed, expected)
            checked += 2
        for query in range(-1, max(capacity, 0) + 2):
            matches = [(request, begin, end) for request, (begin, end) in intervals.items() if begin <= query < end]
            expected = (*matches[0], 1) if len(matches) == 1 else (-1, 0, 0, 0)
            observed = helpers["query_request"](bounds, lengths, query, capacity)
            assert observed == expected, (name, query, observed, expected)
            checked += 1
        for request, length in enumerate(lengths):
            if length > 0:
                continue
            reads.clear()
            helpers["request_bounds"](bounds, lengths, request, capacity)
            assert not any(identity == id(bounds) for identity, _ in reads), (name, "dummy 读取 Q")
        results.append({"case": name, "bounds": bounds, "lengths": lengths, "token_capacity": capacity,
                        "checks": checked, "status": "PASS"})
    output = Path(__file__).with_name("oracle.json")
    output.write_text(json.dumps({"source": str(source), "source_ast_interpreted": True,
                                 "checks": sum(case["checks"] for case in results), "cases": results,
                                 "device_execution": False, "lowered_binary_executed": False},
                                ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {len(results)} cases; {sum(case['checks'] for case in results)} checks; {output}")


if __name__ == "__main__":
    main()
