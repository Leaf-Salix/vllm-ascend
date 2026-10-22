# SPDX-License-Identifier: Apache-2.0
"""B1 policy independence, exact enabled/disabled math and flag forwarding."""

import ast
import copy
import runpy
from pathlib import Path

import pytest
from prepare_output_quant_guard import FLAG
from test_query_resident_weights import functions

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "runs/output_quant_guard"


def test_policy_adds_only_the_single_request_quant_choice():
    old = runpy.run_path(str(HERE / "runs/frozen_v122/policy.py"))["select_policy"]
    new = runpy.run_path(str(CANDIDATE / "policy.py"))["select_policy"]
    for requests in (1, 2, 4, 16, 28, 32):
        for tokens in (1, 6, 7, 8, 9, 24, 48, 56, 96, 144, 168, 192):
            for query in range(1, 7):
                for history in (8192, 32762, 131072):
                    kwargs = dict(tokens=tokens, requests=requests, max_query_tokens=query, max_seq_len=history + query)
                    if tokens > requests * query:
                        for select in (old, new):
                            with pytest.raises(ValueError, match="token容量超过"):
                                select(**kwargs)
                        continue
                    before, after = old(**kwargs).constexprs(), new(**kwargs).constexprs()
                    enabled = after.pop(FLAG)
                    assert after == before
                    assert enabled == (requests == 1 and tokens <= 8)


@pytest.mark.parametrize("enabled,parent", [(True, "output_quant_reuse"), (False, "frozen_v122")])
def test_entire_quant_math_and_tail_are_the_corresponding_measured_branch(enabled, parent):
    old = functions((HERE / "runs" / parent / "output_projection_a_stages.py").read_text())
    new = functions((CANDIDATE / "output_projection_a_stages.py").read_text())
    fn = copy.deepcopy(new["o_projection_groups_quant_stage"])
    assert fn.args.args.pop().arg == FLAG
    assert ast.literal_eval(fn.args.defaults.pop()) is False
    found = 0
    for block in ast.walk(fn):
        body = getattr(block, "body", None)
        if isinstance(body, list):
            for index, node in enumerate(body):
                if isinstance(node, ast.If) and ast.unparse(node.test) == FLAG:
                    body[index : index + 1] = node.body if enabled else node.orelse
                    found += 1
                    break
    assert found == 1
    assert ast.dump(fn, include_attributes=False) == ast.dump(old[fn.name], include_attributes=False)


@pytest.mark.parametrize(
    "filename,targets",
    [
        ("entry.py", {"_csa_core_group": ["vector_schedule"], "decode_csa_spmd": ["_csa_core_group"]}),
        (
            "vector_schedule.py",
            {"vector_schedule": ["o_projection_groups_quant_stage", "o_projection_group_quant_stage"]},
        ),
        ("output_projection_a_stages.py", {"o_projection_group_quant_stage": ["o_projection_groups_quant_stage"]}),
    ],
)
def test_flag_forwarding_keeps_original_schedule(filename, targets):
    old = functions((HERE / "runs/output_quant_reuse" / filename).read_text())
    new = functions((CANDIDATE / filename).read_text())
    for name, callees in targets.items():
        fn = new[name]
        assert fn.args.args.pop().arg == FLAG
        assert ast.literal_eval(fn.args.defaults.pop()) is False
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in callees]
        assert len(calls) == len(callees)
        assert all(ast.unparse(n.args.pop()) == FLAG for n in calls)
        assert ast.dump(fn, include_attributes=False) == ast.dump(old[name], include_attributes=False)
