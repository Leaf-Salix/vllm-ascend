# SPDX-License-Identifier: Apache-2.0
"""SPMD切片的数学提取和所有权门禁，不替代生成代码或NPU验证。"""

import ast
import copy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
OLD = ROOT / "deepseek_v4_flash_csa_tnd_full_precision"
NEW = ROOT / "deepseek_v4_flash_csa_tnd_precision_spmd"


def function(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def test_norm_math_is_preserved():
    before = function(OLD / "input_norm.py", "_input_norm_test")
    after = function(NEW / "norm_wkv_stages.py", "norm_rows")
    old_loop = next(node for node in before.body if isinstance(node, ast.For))
    new_loop = next(node for node in after.body if isinstance(node, ast.For))
    # In an incore helper, reshape the loaded tile rather than the GM tensor.
    # Both views read the same 4096 contiguous BF16 weights in the same order.
    old_load = next(
        node for node in old_loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "w_raw"
    )
    new_load = next(
        node for node in new_loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "w_raw"
    )
    assert ast.unparse(new_load.value) == (
        "pl.reshape(pl.load(weight, [0], [4096], target_memory=pl.MemorySpace.Vec), [1, 4096])"
    )
    new_load.value = copy.deepcopy(old_load.value)
    assert dump(ast.Module(body=old_loop.body, type_ignores=[])) == dump(
        ast.Module(body=new_loop.body, type_ignores=[])
    )
    assert ast.unparse(new_loop.iter) == "pl.range(worker, rows, 48)"


def test_wkv_math_and_rounding_are_preserved():
    before = function(OLD / "wkv.py", "_wkv_test")
    after = function(NEW / "norm_wkv_stages.py", "wkv_matmul")
    old_branch = next(node for node in before.body if isinstance(node, ast.If))
    new_branch = copy.deepcopy(next(node for node in after.body if isinstance(node, ast.If)))
    for old_loop, new_loop in zip(
        (old_branch.body[0], old_branch.orelse[0]), (new_branch.body[0], new_branch.orelse[0])
    ):
        assert ast.unparse(new_loop.iter.args[0]) == "logical_worker"
        assert ast.literal_eval(new_loop.iter.args[2]) == 24
        assert dump(old_loop.iter.args[0]) == dump(new_loop.iter.args[1])
        new_loop.iter = old_loop.iter
    assert dump(old_branch) == dump(new_branch)
    old_round = next(node for node in before.body if isinstance(node, ast.For))
    rounded = function(NEW / "norm_wkv_stages.py", "wkv_round")
    new_round = next(node for node in rounded.body if isinstance(node, ast.For))
    assert dump(ast.Module(body=old_round.body, type_ignores=[])) == dump(
        ast.Module(body=new_round.body, type_ignores=[])
    )
    assert ast.unparse(new_round.iter) == "pl.range(worker, row_tiles, 48)"


@pytest.mark.parametrize("tokens", range(1, 257))
def test_physical_workers_cover_valid_units_once(tokens):
    rows = [row for worker in range(48) for row in range(worker, tokens, 48)]
    assert sorted(rows) == list(range(tokens))
    row_tiles = (tokens + 15) // 16
    units = row_tiles * 32 if tokens <= 16 else ((tokens + 63) // 64) * 16
    work = [unit for worker in range(24) for unit in range(worker, units, 24)]
    assert sorted(work) == list(range(units))
    physical_rows = 16 if tokens <= 16 else ((tokens + 63) // 64) * 64
    assert physical_rows <= tokens + 63
    rounding = [unit for worker in range(48) for unit in range(worker, row_tiles, 48)]
    assert sorted(rounding) == list(range(row_tiles))


def test_all_physical_roles_rendezvous_outside_work_loops():
    for name in ("cube_schedule", "vector_schedule"):
        body = function(NEW / "norm_wkv.py", name).body
        calls = [
            ast.unparse(node.value.func)
            for node in body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        assert calls.count("handoff") == 3
        assert not any(isinstance(node, (ast.If, ast.For)) for node in body)
    tree = ast.parse((NEW / "norm_wkv_stages.py").read_text())
    assert not any(
        isinstance(node, ast.Call) and ast.unparse(node.func) in ("pl.spmd", "pl.at") for node in ast.walk(tree)
    )
    root = function(NEW / "norm_wkv.py", "precision_norm_wkv")
    scopes = [node for node in ast.walk(root) if isinstance(node, ast.With)]
    assert len(scopes) == 1
    assert ast.unparse(scopes[0].items[0].context_expr) == "pl.spmd(GROUPS, sync_start=True)"
    dispatches = [node for node in ast.walk(scopes[0]) if isinstance(node, ast.Call)]
    assert [ast.unparse(node.func) for node in dispatches].count("norm_wkv_group") == 1


def test_cross_role_workspace_contract_matches_kernel_abi():
    for name in ("precision_norm_wkv", "norm_wkv_group"):
        args = function(NEW / "norm_wkv.py", name).args.args
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in args}
        assert annotations["normalized"] == "pl.InOut[pl.Tensor[[T, 4096], pl.BF16]]"
        assert annotations["accum"] == "pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]]"
        assert annotations["accum_read"] == "pl.Tensor[[ACC_ROWS, 512], pl.FP32]"
    group = function(NEW / "norm_wkv.py", "norm_wkv_group")
    calls = [node for node in ast.walk(group) if isinstance(node, ast.Call)]
    vector = next(node for node in calls if ast.unparse(node.func) == "vector_schedule")
    assert ast.unparse(vector.args[4]) == "accum_read"
    cube = next(node for node in calls if ast.unparse(node.func) == "cube_schedule")
    assert ast.unparse(cube.args[3]) == "accum"
