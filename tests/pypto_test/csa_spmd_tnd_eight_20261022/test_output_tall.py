# SPDX-License-Identifier: Apache-2.0
"""Execute candidate ownership code against independent coverage/dependency sets."""

import ast
import collections
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/output_tall_v1"
SIZES = [1, 8, 31, 64, 96, 127, 128, 129, 143, 144, 145, 167, 168, 169, 191, 192]


def oa_geometry(records):
    tree = ast.parse((SOURCE / "output_projection_a_stages.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "o_projection_a_groups_stage_nz")
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None
    loop = next(n for n in node.body if isinstance(n, ast.For))
    stop = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "n0"
    )
    loop.body = loop.body[: stop + 1] + ast.parse("emit(worker_id, g, pa_r0, pa_rows, n0, A_COL_TILE)").body
    namespace = dict(
        pl=SimpleNamespace(range=range, max=max, min=min, tensor=SimpleNamespace(dim=lambda rows, axis: rows)),
        O_GROUPS=8,
        O_LORA=1024,
        emit=lambda *args: records.append(args),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "oa_geometry", "exec"), namespace)
    return namespace[node.name]


def output_schedule(file):
    tree = ast.parse((SOURCE / file).read_text())
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "OUTPUT_GROUP_PIPELINE"
    )
    return compile(ast.Module(body=branch.body, type_ignores=[]), file, "exec")


@pytest.mark.parametrize("tokens", SIZES)
def test_joint_geometry_and_all_producer_dependencies(tokens):
    records = []
    oa = oa_geometry(records)
    cube = output_schedule("cube_schedule.py")
    vector = output_schedule("vector_schedule.py")
    ob_jobs = []
    common = dict(
        tokens=tokens,
        GROUPS=24,
        VECTORS=48,
        CORES=72,
        OA_ROWS=192,
        OA_COLS=128,
        OB_ROWS=128,
        generation=1,
        signal=None,
        pl=SimpleNamespace(range=range),
        PHASE_OA_GROUP_BASE=100,
        PHASE_O_QUANT_GROUP_BASE=200,
        OUTPUT_WORK_BALANCE=True,
        OUTPUT_BALANCE_HEAVY_CORES=16,
        OUTPUT_BALANCE_VIRTUAL_WORKERS=32,
        OUTPUT_BALANCE_LAST_RANK=31,
        OUTPUT_BALANCE_COLUMNS_PER_GROUP=8,
        packed=8 * math.ceil(tokens / 128) * 128,
        wo_a=None,
        o_r=None,
        o_quant=None,
        o_scales=None,
        wo_b=None,
        o_partials=None,
    )
    for worker in range(24):
        quant_acquired = []

        def wait(
            signal,
            phase,
            participants,
            generation,
            first,
            count,
            rows,
            producers,
            base,
            stride,
            *,
            quant_acquired=quant_acquired,
        ):
            assert (participants, generation, rows, producers, base, stride) == (72, 1, 1, 48, 24, 1)
            group = phase - 200
            assert (first, count) == (group * math.ceil(tokens / 32), math.ceil(tokens / 32))
            quant_acquired.append(group)
            return signal

        def ob(
            quant, weights, output, rows, rank, ranks, tile, first, end, *, quant_acquired=quant_acquired, owner=worker
        ):
            assert (rows, ranks, tile, end) == (tokens, 32, 128, first + 1)
            assert quant_acquired.pop() == first
            ob_jobs.extend((job, owner) for job in range(first * 8, end * 8) if job % ranks == rank)
            return output

        namespace = dict(
            common,
            worker=worker,
            wait_row_owners=wait,
            publish_phase=lambda signal, *args: signal,
            o_projection_a_groups_stage_nz=oa,
            o_b_groups_stage=ob,
        )
        exec(cube, namespace)
        assert not quant_acquired
    actual = collections.Counter(
        (group, row, col // 128) for _, group, start, count, col, _ in records for row in range(start, start + count)
    )
    assert actual == collections.Counter(
        {(group, row, col): 1 for group in range(8) for row in range(tokens) for col in range(8)}
    )
    assert sorted(job for job, _ in ob_jobs) == list(range(64))
    counts = collections.Counter(worker for _, worker in ob_jobs)
    assert [counts[w] for w in range(24)] == [2] * 16 + [4] * 8
    oa_counts = collections.Counter(owner for owner, *_ in records)
    # Logical weight loads only: each O_A job reads 1 MiB, each O_B job 0.5 MiB.
    assert [oa_counts[w] * 1024 * 1024 + counts[w] * 512 * 1024 for w in range(24)] == [4 * 1024 * 1024] * 24
    for worker in range(48):
        acquired = []

        def wait(
            signal, phase, participants, generation, first, count, rows, producers, base, stride, *, acquired=acquired
        ):
            assert (participants, generation, rows, producers, base, stride) == (72, 1, 1, 24, 0, 1)
            group = phase - 100
            needed = {owner for owner, g, *_ in records if g == group}
            assert {(first + i) % 24 for i in range(count)} == needed
            acquired.append(group)
            return signal

        def quant(values, output, scales, rows, rank, ranks, tile, first, end, *, acquired=acquired, owner=worker):
            assert (rows, rank, ranks, tile, end) == (tokens, owner, 48, 128, first + 1)
            assert acquired.pop() == first
            return output, scales

        namespace = dict(
            common,
            worker=worker,
            core=24 + worker,
            wait_row_owners=wait,
            o_projection_groups_quant_stage=quant,
            publish_phase=lambda signal, *args: signal,
        )
        exec(vector, namespace)
        assert not acquired


@pytest.mark.parametrize("tokens", [1, 6, 16, 31, 32, 63, 64, 96, 97, 128, 129, 144, 168, 192, 193, 256, 384])
def test_policy_scope_preserves_other_strategies(tokens):
    new = runpy.run_path(str(SOURCE / "policy.py"))["select_policy"]
    old = runpy.run_path(str(HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd/policy.py"))[
        "select_policy"
    ]
    for width in [1, 5, 6]:
        for history in [8192, 32760, 131072, 262144]:
            arguments = dict(
                tokens=tokens, requests=math.ceil(tokens / width), max_query_tokens=width, max_seq_len=history + width
            )
            actual, expected = new(**arguments).constexprs(), old(**arguments).constexprs()
            tall = 128 < tokens <= 192
            assert actual.pop("OA_ROWS") == (192 if tall else 128)
            if tall:
                expected.update(OA_COLS=128, OUTPUT_WORK_BALANCE=True)
            assert actual == expected


def test_quant_and_ob_math_unchanged():
    parent = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    assert (SOURCE / "output_projection_b_stages.py").read_bytes() == (
        parent / "output_projection_b_stages.py"
    ).read_bytes()
    functions = []
    for path in [SOURCE, parent]:
        tree = ast.parse((path / "output_projection_a_stages.py").read_text())
        functions.append(
            [
                ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and "quant" in node.name
            ]
        )
    assert functions[0] == functions[1]
