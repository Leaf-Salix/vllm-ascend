# SPDX-License-Identifier: Apache-2.0
"""Packed workspace ownership and actual SDK inline-output contracts."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
PACKAGE = ROOT / "deepseek_v4_flash_csa_tnd_precision_spmd"
STAGES = PACKAGE / "hc_prefix_indexer_stages.py"


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def test_inline_returns_follow_writable_argument_order():
    # The installed SDK uses Out/InOut declaration order to infer call lhs types.
    # Packing comb previously inverted these two outputs and narrowed arena64 to16.
    for f in functions(STAGES).values():
        writable = [a.arg for a in f.args.args if any(k in ast.unparse(a.annotation) for k in ("pl.Out[", "pl.InOut["))]
        result = f.body[-1]
        assert isinstance(result, ast.Return)
        values = result.value.elts if isinstance(result.value, ast.Tuple) else [result.value]
        assert [ast.unparse(v) for v in values] == writable, f.name


def test_indexer_projection_math_matches_frozen_implementation():
    old = functions(ROOT / "deepseek_v4_flash_csa_tnd_full_precision/indexer_projection.py")["_indexer_projection"]
    loops = [n for n in old.body if isinstance(n, ast.For)]
    for before, name, workers in zip(loops, ("indexer_projection_matmul", "indexer_projection_round"), (24, 48)):
        after = next(n for n in functions(STAGES)[name].body if isinstance(n, ast.For))
        assert ast.dump(ast.Module(body=before.body, type_ignores=[])) == ast.dump(
            ast.Module(body=after.body, type_ignores=[])
        )
        assert ast.unparse(after.iter.args[0]) == "physical_worker" and ast.literal_eval(after.iter.args[2]) == workers


@pytest.mark.parametrize("tokens", range(1, 257))
def test_packed_regions_and_indexer_owners(tokens):
    p16 = (tokens + 15) // 16 * 16
    regions = [
        (range(p16), range(32)),
        (range(p16), range(32, 40)),
        (range(0, p16, 8), range(40, 48)),
        (range(tokens), range(48, 56)),
        (range(p16, p16 + 8), range(8)),
        (range(p16, p16 + 8), range(8, 40)),
    ]
    seen = set()
    for rows, cols in regions:
        addresses = {r * 64 + c for r in rows for c in cols}
        assert addresses and not addresses & seen
        assert min(addresses) >= 0 and max(addresses) < (p16 + 8) * 64
        seen.update(addresses)
    for t0 in range(0, tokens, 8):
        assert t0 + 8 <= p16 and (t0 * 64 + 40) * 4 % 32 == 0
        for lane in range(min(8, tokens - t0)):
            assert (t0 * 64 + 40 + lane) in seen
    tiles = p16 // 16
    for workers, units in ((24, tiles * 4), (48, tiles)):
        assert sorted(i for w in range(workers) for i in range(w, units, workers)) == list(range(units))


def test_p16_split_stride_excludes_tail_arena_rows():
    fs = functions(STAGES)
    for name in ("packed_hc_rms", "packed_hc_linear_reduce"):
        assignment = next(
            n for n in fs[name].body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "t_linear"
        )
        assert ast.unparse(assignment.value) == "pl.tensor.dim(vec_ws, 0) - 8"
    comb = fs["packed_hc_comb"]
    inverse = next(n for n in ast.walk(comb) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "inv_col_t")
    assert ast.unparse(inverse.value.func) == "pl.set_validshape"
    assert [ast.unparse(n) for n in inverse.value.args[1:]] == ["valid_rows", "1"]


def test_one_mixed_group_fits_32_descriptors_and_preserves_handoffs():
    fs = functions(PACKAGE / "hc_prefix_indexer.py")
    root = fs["packed_precision_hc_prefix_indexer"]
    assert len(root.args.args) == 31
    group = fs["packed_prefix_indexer_group"]
    assert len(group.args.args) == 32  # public x replaced by flat; extra hc_base_2d view
    for write, read in [
        ("square", "square_read"),
        ("linear", "linear_read"),
        ("accum", "accum_read"),
        ("idx_accum", "idx_accum_read"),
    ]:
        args = {a.arg: ast.unparse(a.annotation) for a in root.args.args}
        assert args[write] == "pl.InOut[" + args[read] + "]"
    for name in ("packed_prefix_indexer_cube", "packed_prefix_indexer_vector"):
        body = fs[name].body
        barriers = [
            n
            for n in body
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "handoff"
        ]
        assert len(barriers) == 10 and not any(isinstance(n, (ast.If, ast.For)) for n in body)
    scopes = [n for n in ast.walk(root) if isinstance(n, ast.With)]
    assert len(scopes) == 1 and ast.unparse(scopes[0].items[0].context_expr) == "pl.spmd(GROUPS, sync_start=True)"
