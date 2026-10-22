# SPDX-License-Identifier: Apache-2.0
"""Exercise actual scalar ownership code, independent of NPU numeric/performance checks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_control_tiling_candidates import idle_publications


def source_root():
    root = Path(__file__).resolve().parent / "runs/frozen_v122"
    if not root.exists():
        pytest.skip("Materialize the frozen reference first")
    return root


@pytest.mark.parametrize("tokens", [1, 6, 8, 9, 16, 41, 65, 96, 192, 384, 385])
def test_idle_input_planner_still_acquires_fresh_metadata(tokens):
    source = idle_publications((source_root() / "vector_schedule.py").read_text())
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "vector_schedule")
    # Execute the actual new publication block. The real data-writing math is
    # unchanged; its fresh-data boundary is what this test must protect.
    block = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "BALANCED and QUERY_GROUP == 2")
    code = compile(ast.Module(body=[block], type_ignores=[]), "input_publication", "exec")
    active = min((tokens + 7) // 8, 48)
    planner = min((tokens + 7) // 8, 47)
    for query_group, balanced in ((2, True), (2, False), (6, True)):
        for worker in range(48):
            events = []
            namespace = {
                "pl": SimpleNamespace(
                    min=min,
                    system=SimpleNamespace(
                        fence=lambda events=events: events.append("fence"),
                        cacheinvalid=lambda events=events: events.append("acquire"),
                    ),
                ),
                "publish_phase": lambda *args, events=events: events.append("publish"),
                "tokens": tokens,
                "worker": worker,
                "core": 24 + worker,
                "CORES": 72,
                "generation": 7,
                "VECTORS": 48,
                "PHASE_INPUT": 1,
                "signal": None,
                "BALANCED": balanced,
                "QUERY_GROUP": query_group,
            }
            exec(code, namespace)
            if query_group != 2 or not balanced or worker < active:
                assert events == ["publish"]
            elif worker == planner:
                assert events == ["fence", "acquire", "fence"]
            else:
                assert not events


@pytest.mark.parametrize("row_tile", [48, 64])
@pytest.mark.parametrize("tokens", [1, 47, 48, 49, 63, 64, 65, 128, 129, 144, 168, 192, 224, 256])
def test_nz_output_tiles_cover_each_real_row_column_once_and_match_waiters(tokens, row_tile):
    source = (source_root() / "output_projection_a_stages.py").read_text()
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "o_projection_a_groups_stage_nz"
    )
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    writes = []

    class Weight:
        def __getitem__(self, indices):
            group, k, col = indices
            return SimpleNamespace(group=group.start, k=k.start, col=col.start, columns=col.stop - col.start)

    def load_slice(tensor, shape, offsets, valid_shape):
        return SimpleNamespace(rows=valid_shape[0], k=offsets[1])

    def matmul(a, b, out_dtype):
        assert a.k == b.k == 0
        return SimpleNamespace(rows=a.rows, columns=b.columns, ks=[0])

    def matmul_acc(acc, a, b):
        assert a.rows == acc.rows and b.columns == acc.columns and a.k == b.k
        acc.ks.append(a.k)
        return acc

    def assemble(output, acc, offsets):
        assert acc.ks == list(range(0, 4096, 256))
        writes.append((offsets[0], offsets[1], acc.rows, acc.columns))
        return output

    namespace = {
        "pl": SimpleNamespace(
            tensor=SimpleNamespace(dim=lambda t, axis: t.shape[axis]),
            max=max,
            min=min,
            range=range,
            pipeline=lambda *a, stage: range(*a),
            slice=load_slice,
            matmul=matmul,
            matmul_acc=matmul_acc,
            assemble=assemble,
            FP32=None,
        ),
        "PROJ_A_ROW_TILE": row_tile,
        "O_LORA": 1024,
        "O_GROUPS": 8,
        "O_GROUP_IN": 4096,
        "A_K_TILE": 256,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "oa_nz", "exec"), namespace)
    helper = namespace[fn.name]
    owned = {group: set() for group in range(8)}
    covered = set()
    for group in range(8):
        for worker in range(24):
            writes.clear()
            helper(SimpleNamespace(shape=(8 * 256, 4096)), Weight(), None, tokens, worker, 24, 256, group, group + 1)
            if writes:
                owned[group].add(worker)
            for first, column, count, width in writes:
                assert 0 <= first < first + count <= tokens and width == 256
                assert group * 1024 <= column < (group + 1) * 1024 and column % 256 == 0
                for row in range(first, first + count):
                    assert (row, column) not in covered
                    covered.add((row, column))
        tasks = ((tokens + row_tile - 1) // row_tile) * 4
        assert owned[group] == {item % 24 for item in range(group * tasks, (group + 1) * tasks)}
    assert covered == {(row, column) for row in range(tokens) for column in range(0, 8192, 256)}
