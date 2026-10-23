# SPDX-License-Identifier: Apache-2.0
"""Execute actual O_A/O_B owners and verify all column and phase dependencies."""

import ast
import collections
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_output_rectangle as contracts

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/output_group_owners_v2"
PARENT = HERE / "runs/output_rectangle_production_v1"


def actual_oa(records):
    functions = [
        n
        for n in ast.parse((SOURCE / "output_projection_a_stages.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("o_projection_a_groups_stage_nz", "_three_owner_o_projection_a_groups_stage_nz")
    ]
    for fn in functions:
        fn.decorator_list, fn.returns = [], None
        for arg in fn.args.args:
            arg.annotation = None

    class Emit(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node.func) == "_rectangle_o_projection_a_tile_nz":
                return ast.Call(
                    func=ast.Name(id="emit", ctx=ast.Load()),
                    args=[ast.Name(id="worker_id", ctx=ast.Load())] + [node.args[i] for i in (3, 5, 6, 8, 10)],
                    keywords=[],
                )
            return self.generic_visit(node)

    scope = dict(
        pl=SimpleNamespace(min=min, max=max, tensor=SimpleNamespace(dim=lambda value, axis: value)),
        O_GROUPS=8,
        O_LORA=1024,
        emit=lambda *args: records.append(args),
    )
    tree = Emit().visit(ast.Module(body=functions, type_ignores=[]))
    exec(compile(ast.fix_missing_locations(tree), "actual_three_owner", "exec"), scope)
    return scope["o_projection_a_groups_stage_nz"]


@pytest.mark.parametrize("tokens", [1, 15, 16, 31, 32, 63, 64, 65, 80, 95, 96])
def test_exact_outputs_all_waits_and_complementary_ob(tokens, monkeypatch):
    monkeypatch.setattr(contracts, "SOURCE", SOURCE)
    records, ob_jobs = [], []
    oa = actual_oa(records)
    common = dict(
        tokens=tokens,
        GROUPS=24,
        VECTORS=48,
        CORES=72,
        OA_ROWS=96,
        OA_COLS=256,
        OB_ROWS=96,
        generation=1,
        signal=None,
        pl=SimpleNamespace(range=range),
        PHASE_OA_GROUP_BASE=100,
        PHASE_O_QUANT_GROUP_BASE=200,
        OUTPUT_WORK_BALANCE=True,
        packed=8 * 128,
        wo_a=None,
        o_r=None,
        o_quant=None,
        o_scales=None,
        wo_b=None,
        o_partials=None,
    )
    for worker in range(24):
        acquired = []

        def wait(_signal, phase, participants, gen, first, count, rows, producers, base, stride, *, acquired=acquired):
            assert (participants, gen, rows, producers, base, stride) == (72, 1, 1, 48, 24, 1)
            group = phase - 200
            assert (first, count) == (group * math.ceil(tokens / 32), math.ceil(tokens / 32))
            acquired.append(group)

        def ob(_q, _w, out, rows, rank, ranks, tile, first, end, *, acquired=acquired, owner=worker):
            assert (rows, ranks, tile, end) == (tokens, 24, 96, first + 1)
            assert acquired.pop() == first
            ob_jobs.extend((job, owner) for job in range(first * 8, end * 8) if job % ranks == rank)
            return out

        exec(
            contracts.schedule("cube_schedule.py"),
            dict(
                common,
                worker=worker,
                wait_row_owners=wait,
                publish_phase=lambda s, *_: s,
                o_projection_a_groups_stage_nz=oa,
                o_b_groups_stage=ob,
            ),
        )
        assert not acquired
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
    assert collections.Counter(r[-1] for r in records) == {256: 24, 64: 8, 96: 16}
    assert collections.Counter(r[0] for r in records) == {core: 2 for core in range(24)}
    assert sorted(job for job, _ in ob_jobs) == list(range(64))
    ob_counts = collections.Counter(owner for _, owner in ob_jobs)
    work_classes = collections.Counter()
    for worker in range(24):
        columns = sum(r[-1] for r in records if r[0] == worker)
        work_classes[columns, ob_counts[worker]] += 1
    assert work_classes == {(320, 3): 8, (352, 3): 8, (352, 2): 8}
    for worker in range(48):
        acquired = []

        def wait(_signal, phase, participants, gen, first, count, rows, producers, base, stride, *, acquired=acquired):
            assert (participants, gen, rows, producers, base, stride) == (72, 1, 1, 24, 0, 1)
            group = phase - 100
            assert {(first + i) % 24 for i in range(count)} == {owner for owner, g, *_ in records if g == group}
            assert count == 3
            acquired.append(group)

        def quant(_v, out, scales, rows, rank, ranks, tile, first, end, *, acquired=acquired, owner=worker):
            assert (rows, rank, ranks, tile, end) == (tokens, owner, 48, 96, first + 1)
            assert acquired.pop() == first
            return out, scales

        exec(
            contracts.schedule("vector_schedule.py"),
            dict(
                common,
                worker=worker,
                core=24 + worker,
                wait_row_owners=wait,
                o_projection_groups_quant_stage=quant,
                publish_phase=lambda s, *_: s,
            ),
        )
        assert not acquired


@pytest.mark.parametrize("tokens", [1, 6, 32, 48, 56, 64, 65, 95, 96, 97, 128, 129, 144, 160, 161, 168, 192])
def test_only_bounded_oa_resource_policy_changes(tokens):
    new, old = [runpy.run_path(str(root / "policy.py"))["select_policy"] for root in (SOURCE, PARENT)]
    for width in (1, 5, 6):
        for history in (8192, 32760, 131072, 262144):
            kwargs = dict(
                tokens=tokens, requests=math.ceil(tokens / width), max_query_tokens=width, max_seq_len=history + width
            )
            actual, expected = new(**kwargs).constexprs(), old(**kwargs).constexprs()
            if (history + width) // 4 > 8192 and expected["QUERY_GROUP"] == 6 and 64 < tokens <= 96:
                expected.update(OA_ROWS=96, OA_COLS=256)
            assert actual == expected


def test_matrix_chain_quant_ob_and_unrelated_modules_preserved():
    changed = {"policy.py", "cube_schedule.py", "vector_schedule.py", "output_projection_a_stages.py"}
    for path in PARENT.glob("*.py"):
        if path.name not in changed:
            assert path.read_bytes() == (SOURCE / path.name).read_bytes()
    for function in ("_rectangle_o_projection_a_tile_nz", "o_projection_groups_quant_stage"):
        bodies = []
        for root in (SOURCE, PARENT):
            tree = ast.parse((root / "output_projection_a_stages.py").read_text())
            bodies.append(ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)))
        assert bodies[0] == bodies[1]
