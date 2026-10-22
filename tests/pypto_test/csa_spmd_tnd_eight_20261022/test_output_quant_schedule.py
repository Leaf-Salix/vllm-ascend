# SPDX-License-Identifier: Apache-2.0
"""Execute actual ownership loops; compare row coverage with independent sets."""

import ast
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SIZES = [0, 1, 6, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 95, 96, 97,
         127, 128, 129, 143, 144, 145, 167, 168, 169, 191, 192, 193, 255, 256, 257]


def ownership_function(variant, nz):
    source = (HERE / "runs" / variant / "output_projection_a_stages.py").read_text()
    tree = ast.parse(source)
    task_tile = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "QUANT_TASK_T_TILE")
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "o_projection_groups_quant_stage")
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None
    for loop in ast.walk(node):
        if isinstance(loop, ast.For) and isinstance(loop.target, ast.Name):
            if loop.target.id == "qt":
                loop.body = ast.parse('rows = min(8, tokens - qt)\nemit("active", worker_id, g, qt, rows)').body
            elif loop.target.id == "zt":
                loop.body = ast.parse('rows = min(8, padded_rows - zt)\nemit("zero", worker_id, g, zt, rows)').body
    records = []
    namespace = dict(
        pl=SimpleNamespace(range=range, pipeline=lambda *args, **kwargs: range(*args), min=min),
        QUANT_TASK_T_TILE=task_tile, QUANT_TOKEN_TILE=8, O_LORA=1024,
        QUANT_WEIGHT_NZ=nz, PROJ_B_MEDIUM_T_TILE=96,
        emit=lambda *args: records.append(args),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), source[:0], "exec"), namespace)
    return namespace[node.name], records, task_tile


@pytest.mark.parametrize("variant", ["output_quant_task16_v1", "output_quant_no_unused_pad_v1"])
@pytest.mark.parametrize("nz", [False, True])
@pytest.mark.parametrize("row_tile", [32, 96, 128])
def test_all_rows_unique_and_padding_only_when_consumed(variant, nz, row_tile):
    function, records, _ = ownership_function(variant, nz)
    for tokens in SIZES:
        records.clear()
        for group in range(8):
            for worker in range(48):
                function(None, None, None, tokens, worker, 48, row_tile, group, group + 1)
        active = Counter((g, row) for kind, _, g, begin, count in records if kind == "active"
                         for row in range(begin, begin + count))
        zero = Counter((g, row) for kind, _, g, begin, count in records if kind == "zero"
                       for row in range(begin, begin + count))
        padded = math.ceil(tokens / row_tile) * row_tile
        if variant == "output_quant_no_unused_pad_v1" and nz and row_tile > 96:
            padded = tokens
        assert active == Counter({(g, row): 1 for g in range(8) for row in range(tokens)})
        assert zero == Counter({(g, row): 1 for g in range(8) for row in range(tokens, padded)})
        for group in range(8):
            tail_writers = {worker for kind, worker, g, _, _ in records if kind == "zero" and g == group}
            assert len(tail_writers) <= 1


def test_cube_and_vector_dependencies_match_new_owners():
    function, records, tile = ownership_function("output_quant_task16_v1", True)
    expressions = []
    for file in ["cube_schedule.py", "vector_schedule.py"]:
        tree = ast.parse((HERE / "runs/output_quant_task16_v1" / file).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "quant_per_group":
                    expressions.append(compile(ast.Expression(node.value), file, "eval"))
    assert len(expressions) == 2
    for tokens in SIZES:
        records.clear()
        for worker in range(48):
            function(None, None, None, tokens, worker, 48, 128, 0, 8)
        for expression in expressions:
            count = eval(expression, dict(tokens=tokens, QUANT_TASK_T_TILE=tile))
            for group in range(8):
                waited = {(group * count + i) % 48 for i in range(count)}
                actual = {worker for _, worker, g, _, _ in records if g == group}
                assert waited == actual


def test_quant_math_and_both_o_b_consumers_unchanged():
    base = HERE / "runs/output_quant_task16_v1"
    tail = HERE / "runs/output_quant_no_unused_pad_v1"
    assert (base / "output_projection_b_stages.py").read_bytes() == (
        tail / "output_projection_b_stages.py"
    ).read_bytes()
    def math_body(path):
        tree = ast.parse(path.read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "qt")
        return [ast.dump(n, include_attributes=False) for n in loop.body]
    assert math_body(base / "output_projection_a_stages.py") == math_body(tail / "output_projection_a_stages.py")
