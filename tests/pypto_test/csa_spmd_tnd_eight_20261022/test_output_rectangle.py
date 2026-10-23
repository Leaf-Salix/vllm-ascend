# SPDX-License-Identifier: Apache-2.0
"""Execute rectangle ownership and waits, including the non-dividing N tail."""

import ast
import collections
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/output_rectangle_v3"
PARENT = HERE / "runs/physical_capacity_production_v1"


def geometry(records):
    tree = ast.parse((SOURCE / "output_projection_a_stages.py").read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rectangle_o_projection_a_groups_stage_nz"
    )
    fn.decorator_list, fn.returns = [], None
    for arg in fn.args.args:
        arg.annotation = None

    class CaptureTile(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node.func) == "_rectangle_o_projection_a_tile_nz":
                return ast.Call(
                    func=ast.Name(id="emit", ctx=ast.Load()),
                    args=[ast.Name(id="worker_id", ctx=ast.Load())] + [node.args[i] for i in (3, 5, 6, 8, 10)],
                    keywords=[],
                )
            return self.generic_visit(node)

    fn = CaptureTile().visit(fn)
    scope = dict(
        pl=SimpleNamespace(range=range, min=min, max=max, tensor=SimpleNamespace(dim=lambda x, _axis: x)),
        O_GROUPS=8,
        O_LORA=1024,
        emit=lambda *args: records.append(args),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "oa_geometry", "exec"), scope)
    return scope[fn.name]


def schedule(filename):
    tree = ast.parse((SOURCE / filename).read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "OUTPUT_GROUP_PIPELINE")
    return compile(ast.Module(body=branch.body, type_ignores=[]), filename, "exec")


@pytest.mark.parametrize("tokens", [1, 7, 16, 64, 127, 128, 129, 143, 144, 145, 159, 160])
def test_exact_columns_and_dependency_coverage(tokens):
    records, ob_jobs = [], []
    oa = geometry(records)
    common = dict(
        tokens=tokens,
        GROUPS=24,
        VECTORS=48,
        CORES=72,
        OA_ROWS=160,
        OA_COLS=192,
        OB_ROWS=128,
        generation=1,
        signal=None,
        pl=SimpleNamespace(range=range),
        PHASE_OA_GROUP_BASE=100,
        PHASE_O_QUANT_GROUP_BASE=200,
        OUTPUT_WORK_BALANCE=False,
        packed=8 * math.ceil(tokens / 128) * 128,
        wo_a=None,
        o_r=None,
        o_quant=None,
        o_scales=None,
        wo_b=None,
        o_partials=None,
    )
    for worker in range(24):
        acquired = []

        def wait(
            _signal, phase, participants, generation, first, count, rows, producers, base, stride, *, acquired=acquired
        ):
            assert (participants, generation, rows, producers, base, stride) == (72, 1, 1, 48, 24, 1)
            group = phase - 200
            assert (first, count) == (group * math.ceil(tokens / 32), math.ceil(tokens / 32))
            acquired.append(group)

        def ob(_q, _w, out, rows, rank, ranks, tile, first, end, *, acquired=acquired, owner=worker):
            assert (rows, rank, ranks, tile, end) == (tokens, owner, 24, 128, first + 1)
            assert acquired.pop() == first
            ob_jobs.extend((job, owner) for job in range(first * 8, end * 8) if job % ranks == rank)
            return out

        exec(
            schedule("cube_schedule.py"),
            dict(
                common,
                worker=worker,
                wait_row_owners=wait,
                publish_phase=lambda signal, *_args: signal,
                o_projection_a_groups_stage_nz=oa,
                o_b_groups_stage=ob,
            ),
        )
        assert not acquired
    # Compare independently enumerated 16-column units; the final N64 contributes
    # four units rather than twelve and cannot overwrite the following group.
    actual = collections.Counter(
        (group, row, col // 16)
        for _, group, first, rows, col0, cols in records
        for row in range(first, first + rows)
        for col in range(col0, col0 + cols, 16)
    )
    assert actual == collections.Counter(
        {(g, row, col): 1 for g in range(8) for row in range(tokens) for col in range(64)}
    )
    assert len(records) == 48
    assert collections.Counter(record[-1] for record in records) == {192: 40, 64: 8}
    assert collections.Counter(record[0] for record in records) == {core: 2 for core in range(24)}
    assert sorted(job for job, _ in ob_jobs) == list(range(64))
    assert all(job % 24 == core for job, core in ob_jobs)
    for worker in range(48):
        acquired = []

        def wait(
            _signal, phase, participants, generation, first, count, rows, producers, base, stride, *, acquired=acquired
        ):
            assert (participants, generation, rows, producers, base, stride) == (72, 1, 1, 24, 0, 1)
            group = phase - 100
            assert {(first + i) % 24 for i in range(count)} == {owner for owner, g, *_ in records if g == group}
            acquired.append(group)

        def quant(_v, out, scales, rows, rank, ranks, tile, first, end, *, acquired=acquired, owner=worker):
            assert (rows, rank, ranks, tile, end) == (tokens, owner, 48, 128, first + 1)
            assert acquired.pop() == first
            return out, scales

        exec(
            schedule("vector_schedule.py"),
            dict(
                common,
                worker=worker,
                core=24 + worker,
                wait_row_owners=wait,
                o_projection_groups_quant_stage=quant,
                publish_phase=lambda signal, *_args: signal,
            ),
        )
        assert not acquired


@pytest.mark.parametrize("tokens", [1, 6, 31, 64, 96, 97, 128, 129, 144, 159, 160, 161, 168, 192, 193, 256])
def test_only_rectangle_resource_domain_changes(tokens):
    new, old = [runpy.run_path(str(root / "policy.py"))["select_policy"] for root in (SOURCE, PARENT)]
    for width in (1, 5, 6):
        for history in (8192, 32760, 131072, 262144):
            kw = dict(
                tokens=tokens, requests=math.ceil(tokens / width), max_query_tokens=width, max_seq_len=history + width
            )
            actual, expected = new(**kw).constexprs(), old(**kw).constexprs()
            active = 128 < tokens <= 160
            assert actual.pop("OA_ROWS") == (160 if active else 128)
            if active:
                expected["OA_COLS"] = 192
            assert actual == expected


def test_k_chain_quant_and_ob_preserved():
    assert (SOURCE / "output_projection_b_stages.py").read_bytes() == (
        PARENT / "output_projection_b_stages.py"
    ).read_bytes()
    trees = [ast.parse((root / "output_projection_a_stages.py").read_text()) for root in (SOURCE, PARENT)]
    quant = [
        [ast.dump(fn) for fn in tree.body if isinstance(fn, ast.FunctionDef) and "quant" in fn.name] for tree in trees
    ]
    assert quant[0] == quant[1]
    for name in ("o_projection_a_stage_nd", "o_projection_a_groups_stage_nz"):
        candidate_name = "_rectangle_o_projection_a_tile_nz" if name.endswith("_nz") else "_rectangle_" + name
        functions = [
            next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == selected)
            for tree, selected in zip(trees, (candidate_name, name))
        ]
        calls = [
            [
                ast.dump(n)
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pl.matmul", "pl.matmul_acc", "pl.pipeline")
            ]
            for fn in functions
        ]
        assert calls[0] == calls[1]
