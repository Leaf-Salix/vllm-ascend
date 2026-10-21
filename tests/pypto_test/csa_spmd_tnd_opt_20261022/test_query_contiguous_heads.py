# SPDX-License-Identifier: Apache-2.0
"""Check real tile ownership and original per-head arithmetic before hardware."""

import ast
from pathlib import Path

from prepare_query_contiguous_heads import transform

HERE = Path(__file__).resolve().parent


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def candidate_loop():
    source = (HERE / "runs/frozen_v122/query_projection_stages.py").read_text()
    fn = functions(transform(source))["_q_dequant_rope_contiguous_heads"]
    return next(n for n in fn.body if isinstance(n, ast.For))


def test_token_tail_is_entire_original_helper_and_other_functions_unchanged():
    source = (HERE / "runs/frozen_v122/query_projection_stages.py").read_text()
    old = functions(source)
    new = functions(transform(source))
    retained = new["_q_dequant_rope_original"]
    retained.name = "q_dequant_rope_stage"
    assert ast.dump(retained, include_attributes=False) == ast.dump(old[retained.name], include_attributes=False)
    for name in old:
        if name != retained.name:
            assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)
    branch = new["q_dequant_rope_stage"].body[0]
    assert ast.unparse(branch.test) == "pl.tensor.dim(q_flat, 0) % Q_ROPE_T_TILE == 0"
    assert ast.unparse(branch.orelse[0].value.func) == "_q_dequant_rope_original"


def test_real_head_mapping_has_one_owner_and_preserves_affine_native_addresses():
    loop = candidate_loop()
    exprs = {n.targets[0].id: n.value for n in loop.body if isinstance(n, ast.Assign)}

    def value(expr, ns):
        return eval(compile(ast.Expression(expr), "<source-index>", "eval"), {}, ns)

    for tokens in (0, 8, 16, 24, 48, 56, 64, 96, 144, 168, 192, 256, 384):
        for workers in (24, 32, 48):
            counts = {}
            for worker in range(workers):
                for work in range(worker, tokens * 8, workers):
                    ns = dict(H=64, Q_ROPE_T_TILE=8, dq_work=work)
                    for name in ("qt", "hg", "head_row"):
                        ns[name] = value(exprs[name], ns)
                    for lane in range(8):
                        token, head = ns["qt"], ns["hg"] + lane
                        assert 0 <= token < tokens and 0 <= head < 64
                        assert (ns["head_row"] + lane) * 512 == token * (64 * 512) + head * 512
                        counts[token, head] = counts.get((token, head), 0) + 1
            assert len(counts) == tokens * 64 and all(count == 1 for count in counts.values())


def test_per_row_rms_and_rope_math_preserve_operations_and_rounding():
    source = (HERE / "runs/frozen_v122/query_projection_stages.py").read_text()
    old = functions(source)["q_dequant_rope_stage"]
    original_loop = next(n for n in ast.walk(old) if isinstance(n, ast.For) and ast.unparse(n.target) == "h_inner")
    new_loop = candidate_loop()

    def arithmetic(loop):
        assignments = {n.targets[0].id: n.value for n in loop.body if isinstance(n, ast.Assign)}
        del assignments["q_head_acc"]
        del assignments["q_head_scale"]
        return {
            name: ast.dump(value, include_attributes=False)
            for name, value in assignments.items()
            if name.startswith(("q_head_", "q_nope_", "q_rope_", "q_swapped_flat"))
        }

    expected = arithmetic(original_loop)
    actual = arithmetic(new_loop)
    assert actual.pop("q_head_row_scaled") == ast.dump(
        ast.parse("pl.mul(q_head_acc_fp32, qr_scale_scalar)", mode="eval").body, include_attributes=False
    )
    assert expected.pop("q_head_row_scaled") == ast.dump(
        ast.parse("pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)", mode="eval").body, include_attributes=False
    )
    scalar = next(
        n for n in new_loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "qr_scale_scalar"
    )
    assert ast.unparse(scalar.value) == "pl.read(qr_scale_pad_store, [qt, 0])"
    # Weight scale has eight distinct head rows instead of one broadcast head.
    # Activation scale is still applied first, before this second FP32 product.
    assert actual.pop("q_head_dq") == ast.dump(
        ast.parse("pl.mul(q_head_row_scaled, q_head_scale)", mode="eval").body, include_attributes=False
    )
    assert expected.pop("q_head_dq") == ast.dump(
        ast.parse("pl.col_expand_mul(q_head_row_scaled, q_head_scale)", mode="eval").body, include_attributes=False
    )
    assert actual == expected
