# SPDX-License-Identifier: Apache-2.0
"""HC prefix math/ownership contracts; actual floating-point proof needs NPU."""

import ast
import copy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
OLD = ROOT / "deepseek_v4_flash_csa_tnd_full_precision/hc_pre.py"
NEW = ROOT / "deepseek_v4_flash_csa_tnd_precision_spmd"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def loop_body(node):
    return ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)


@pytest.mark.parametrize(
    ("new_name", "old_name", "target", "workers"),
    [
        ("hc_widen", "_hc_pre_test", "row", 48),
        ("hc_square", "hc_pre_gates", "sq_unit", 24),
        ("hc_rms", "hc_pre_gates", "rms_block", 48),
        ("hc_pack_weight", "hc_pre_gates_from_rms", "panel64_block", 48),
        ("hc_pack_input", "hc_pre_gates_from_rms", "panel64_input_group", 48),
        ("hc_linear_reduce", "hc_pre_gates_from_rms", "linear_block", 48),
        ("hc_comb", "hc_pre_gates_from_rms", "ob", 48),
        ("hc_mix", "_hc_pre", "blk", 48),
    ],
)
def test_hc_unit_math_unchanged(new_name, old_name, target, workers):
    before = function(OLD, old_name)
    after = function(NEW / "hc_prefix_stages.py", new_name)
    old_loop = next(n for n in ast.walk(before) if isinstance(n, ast.For) and ast.unparse(n.target) == target)
    new_loop = next(n for n in ast.walk(after) if isinstance(n, ast.For) and ast.unparse(n.target) == target)
    assert loop_body(old_loop) == loop_body(new_loop)
    assert ast.unparse(new_loop.iter.args[0]) == "worker"
    assert ast.literal_eval(new_loop.iter.args[2]) == workers


def test_hc_linear_branches_and_prepost_math_unchanged():
    before = function(OLD, "hc_pre_gates_from_rms")
    old_branch = next(
        n
        for n in before.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "t_dim >= 64" and isinstance(n.body[0], ast.For)
    )
    after = function(NEW / "hc_prefix_stages.py", "hc_linear")
    new_branch = copy.deepcopy(next(n for n in after.body if isinstance(n, ast.If)))
    for a, b in zip((old_branch.body[0], old_branch.orelse[0]), (new_branch.body[0], new_branch.orelse[0])):
        assert ast.unparse(b.iter.args[0]) == "worker" and ast.literal_eval(b.iter.args[2]) == 24
        b.iter = a.iter
    assert ast.dump(old_branch) == ast.dump(new_branch)
    old_loop = next(n for n in ast.walk(before) if isinstance(n, ast.For) and ast.unparse(n.target) == "ob_worker")
    old_inner = next(n for n in ast.walk(old_loop) if isinstance(n, ast.For) and ast.unparse(n.target) == "ob")
    new_pre = function(NEW / "hc_prefix_stages.py", "hc_pre_post")
    new_loop = next(n for n in new_pre.body if isinstance(n, ast.For))
    assert loop_body(old_inner) == loop_body(new_loop)
    assert ast.unparse(new_loop.iter) == "pl.range(worker, token_tiles, 48)"


@pytest.mark.parametrize("tokens", range(1, 257))
def test_prefix_unit_ownership_and_partial_capacity(tokens):
    p16 = (tokens + 15) // 16 * 16
    q8 = (tokens + 7) // 8
    group = 4 if tokens >= 64 else 1
    vector_units = [tokens, q8, 64, p16 // 16 * 64 // (4 if tokens >= 64 else 1), p16 // 16, q8, q8, q8]
    cube_units = [p16 // 16 * ((22 + group - 1) // group), 22 * ((p16 + 63) // 64) if tokens >= 64 else 22 * p16 // 16]
    for workers, units in [(48, u) for u in vector_units] + [(24, u) for u in cube_units]:
        assert sorted(i for w in range(workers) for i in range(w, units, workers)) == list(range(units))
    assert q8 * 8 <= p16
    old_pre = [i for w in range(min(q8, 16)) for i in range(w, q8, min(q8, 16))]
    assert sorted(old_pre) == list(range(q8))
    for split in range(22):
        for row in range(0, p16, 64):
            assert split * p16 + row + min(64, p16 - row) <= (split + 1) * p16


def test_prefix_one_spmd_and_unconditional_role_handoffs():
    for role in ("prefix_cube", "prefix_vector"):
        body = function(NEW / "hc_prefix.py", role).body
        direct = [
            n
            for n in body
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "handoff"
        ]
        assert len(direct) == 10
        assert not any(isinstance(n, (ast.If, ast.For)) for n in body)
    root = function(NEW / "hc_prefix.py", "precision_hc_prefix")
    scopes = [n for n in ast.walk(root) if isinstance(n, ast.With)]
    assert len(scopes) == 1 and ast.unparse(scopes[0].items[0].context_expr) == "pl.spmd(GROUPS, sync_start=True)"
    args = {n.arg: ast.unparse(n.annotation) for n in root.args.args}
    assert len(args) == 29
    for write, read in [("square", "square_read"), ("linear", "linear_read"), ("accum", "accum_read")]:
        assert args[write] == "pl.InOut[" + args[read] + "]"
    stages = ast.parse((NEW / "hc_prefix_stages.py").read_text())
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) in ("pl.spmd", "pl.at") for n in ast.walk(stages))
