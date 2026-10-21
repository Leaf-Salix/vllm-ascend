# SPDX-License-Identifier: Apache-2.0
"""Guard range, independent policy, preserved math, and complete flag plumbing."""

import ast
import dataclasses
import runpy
from pathlib import Path

import pytest
from prepare_query_head_guard import FLAG

HERE = Path(__file__).resolve().parent
PARENT = HERE / "runs/query_contiguous_heads_v4"
CANDIDATE = HERE / "runs/query_head_guard"


def funcs(root, name):
    return {n.name: n for n in ast.parse((root / name).read_text()).body if isinstance(n, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


def test_policy_only_adds_the_independent_long_g6_resource_guard():
    old = runpy.run_path(str(PARENT / "policy.py"))["select_policy"]
    new = runpy.run_path(str(CANDIDATE / "policy.py"))["select_policy"]
    for tokens in (1, 6, 24, 32, 48, 56, 63, 64, 65, 72, 80, 88, 95, 96, 97, 144, 168, 192):
        for requests in (1, 4, 16, 32):
            for query in (1, 4, 5, 6):
                for history in (8192, 32762, 32768, 131072):
                    kwargs = dict(tokens=tokens, requests=requests, max_query_tokens=query, max_seq_len=history + query)
                    if tokens > requests * query:
                        for select in (old, new):
                            with pytest.raises(ValueError, match="token容量超过"):
                                select(**kwargs)
                        continue
                    before, after = old(**kwargs).constexprs(), new(**kwargs).constexprs()
                    enabled = after.pop(FLAG)
                    assert after == before
                    assert enabled == (
                        (history + query) // 4 > 8192 and before["QUERY_GROUP"] == 6 and 64 <= tokens <= 96
                    )
    policy = new(tokens=96, requests=16, max_query_tokens=6, max_seq_len=131078)
    assert dataclasses.replace(policy, local_attention_stats=False, plan_physical_rows=False).constexprs()[FLAG]


def test_enabled_wrapper_is_measured_v4_and_disabled_calls_the_entire_original():
    old, new = funcs(PARENT, "query_projection_stages.py"), funcs(CANDIDATE, "query_projection_stages.py")
    for name in old:
        if name != "q_dequant_rope_stage":
            assert dump(new[name]) == dump(old[name])
    wrapper = new["q_dequant_rope_stage"]
    assert wrapper.args.args[-1].arg == FLAG and ast.literal_eval(wrapper.args.defaults[-1]) is False
    guard = wrapper.body[0]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == FLAG
    assert dump(guard.body[0]) == dump(old[wrapper.name].body[0])
    assert ast.unparse(guard.orelse[0].value.func) == "_q_dequant_rope_original"
    assert [ast.unparse(n) for n in guard.orelse[0].value.args] == [n.arg for n in old[wrapper.name].args.args]
    assert dump(wrapper.body[1]) == dump(old[wrapper.name].body[1])


@pytest.mark.parametrize(
    "filename,targets",
    [
        ("entry.py", {"_csa_core_group": "vector_schedule", "decode_csa_spmd": "_csa_core_group"}),
        ("vector_schedule.py", {"vector_schedule": "q_dequant_rope_stage"}),
    ],
)
def test_flag_is_forwarded_without_changing_existing_schedule(filename, targets):
    old, new = funcs(PARENT, filename), funcs(CANDIDATE, filename)
    for name, target in targets.items():
        fn = new[name]
        assert fn.args.args.pop().arg == FLAG
        assert ast.literal_eval(fn.args.defaults.pop()) is False
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == target]
        assert len(calls) == 1 and ast.unparse(calls[0].args.pop()) == FLAG
        assert dump(fn) == dump(old[name])
