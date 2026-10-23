# SPDX-License-Identifier: Apache-2.0
"""Check the critical-chain placement, physical core ownership and old consumers."""

import ast
import collections
from types import SimpleNamespace

import pytest
import test_joint_prefix as contracts

SOURCE = contracts.HERE / "runs/critical_prefix_v1"
WIDTHS = {"main_value": 1024, "main_score": 1024, "kv_fp32": 512}


def grid(tokens, row_tile):
    tree = ast.parse((SOURCE / "joint_prefix_stages.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "joint_bf16_prefix_stage")
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None
    records = []
    namespace = dict(pl=SimpleNamespace(range=range, min=min, tensor=SimpleNamespace(dim=lambda x, _: x)))
    for definition in tree.body:
        if isinstance(definition, ast.Assign) and isinstance(definition.value, ast.Constant):
            namespace[definition.targets[0].id] = definition.value.value
    mapping = {"main_wkv": "main_value", "main_wgate": "main_score", "wkv": "kv_fp32"}

    def emit(x, weight, output, row, column, rows, tile, transpose):
        assert x == tokens and tile == row_tile and mapping[weight] == output
        assert transpose == (weight != "wkv")
        records.append((namespace["owner"], output, row, rows, column))
        return output

    namespace["_project_main"] = lambda *args: emit(*args, True)
    namespace["_project_forward"] = lambda *args: emit(*args, False)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "actual_critical_grid", "exec"),
        namespace,
    )
    for worker in range(24):
        namespace["owner"] = worker
        result = namespace[node.name](
            tokens, "main_wkv", "main_wgate", "main_value", "main_score", "wkv", "kv_fp32", worker, 24, row_tile
        )
        assert result == tuple(WIDTHS)
    return records


@pytest.mark.parametrize(
    "tokens", [0, 1, 6, 32, 64, 65, 80, 95, 96, 127, 128, 129, 143, 144, 167, 168, 191, 192, 193, 256]
)
@pytest.mark.parametrize("row_tile", [128, 192])
def test_each_real_output_rectangle_has_exactly_one_writer(tokens, row_tile):
    records = grid(tokens, row_tile)
    actual = collections.Counter(
        (output, row, column // 128) for _, output, first, rows, column in records for row in range(first, first + rows)
    )
    expected = collections.Counter(
        {
            (name, row, column): 1
            for name, width in WIDTHS.items()
            for row in range(tokens)
            for column in range(width // 128)
        }
    )
    assert actual == expected
    if 0 < tokens <= row_tile:
        jobs = collections.Counter(owner for owner, *_ in records)
        assert [jobs[owner] for owner in range(24)] == [0] * 4 + [1] * 20


@pytest.mark.parametrize("tokens", [65, 80, 96, 127, 128, 129, 144, 168, 192])
def test_existing_consumers_acquire_every_new_writer(tokens, monkeypatch):
    monkeypatch.setattr(contracts, "SOURCE", SOURCE)
    monkeypatch.setattr(contracts, "grid", grid)
    contracts.test_existing_consumers_acquire_every_new_writer(tokens)


@pytest.mark.parametrize("tokens", [1, 6, 16, 32, 48, 56, 64, 65, 96, 128, 129, 144, 168, 192, 193, 256])
def test_policy_changes_only_joint_projection_controls(tokens, monkeypatch):
    monkeypatch.setattr(contracts, "SOURCE", SOURCE)
    contracts.test_policy_changes_only_joint_projection_controls(tokens)


def test_early_indexer_is_original_and_main_kv_only_publish_once():
    cube = (SOURCE / "cube_schedule.py").read_text()
    original = (contracts.PRODUCTION / "cube_schedule.py").read_text()
    start = "    qa = q_a_stage("
    end = "    signal = publish_phase(signal, PHASE_IDX_QUERY_PROJECT, worker, CORES, generation)\n"
    assert (
        cube[cube.index(start) : cube.index(end) + len(end)]
        == original[original.index(start) : original.index(end) + len(end)]
    )
    tree = ast.parse(cube)
    joint = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "JOINT_BF16_PREFIX and COOPERATIVE_PREFIX"
    )
    phases = [
        n.value.args[1].id
        for n in joint.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "publish_phase"
    ]
    assert phases == ["PHASE_MAIN_PROJECT", "PHASE_KV_PROJECT"]
    assert cube.index(end) < cube.index("    if JOINT_BF16_PREFIX") < cube.index(" = projection_overlap_stage(")
    # The previously checked off guards still suppress both legacy producers.
    assert (SOURCE / "projection_overlap_stages.py").read_bytes() == (
        contracts.HERE / "runs/joint_prefix_v2/projection_overlap_stages.py"
    ).read_bytes()


def test_native_bindings_math_and_sync_are_preserved(monkeypatch):
    monkeypatch.setattr(contracts, "SOURCE", SOURCE)
    contracts.test_native_bindings_and_other_math_are_preserved()
    old = ast.parse((contracts.HERE / "runs/joint_prefix_v2/joint_prefix_stages.py").read_text())
    new = ast.parse((SOURCE / "joint_prefix_stages.py").read_text())
    for name in ("_project_main", "_project_forward"):
        a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b)
