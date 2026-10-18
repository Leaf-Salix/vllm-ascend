# SPDX-License-Identifier: Apache-2.0
"""直接执行原DQ源码的Scalar前缀，核对区间事实；不估算性能、不运行NPU。"""

import ast
import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    original = ast.parse((here / "source_reference/query_projection_stages.py").read_text())
    function = next(
        node for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "q_dequant_rope_stage"
    )
    prefix = []
    for statement in function.body:
        if isinstance(statement, ast.For):
            break
        prefix.append(statement)
    code = compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), "original_dq_prefix", "exec")
    spec = importlib.util.spec_from_file_location("region_policy_validation", here / "source_candidate/policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for tokens in range(1, 4097):
        context = {
            "pl": types.SimpleNamespace(tensor=types.SimpleNamespace(dim=lambda tensor, axis: tokens)),
            "q_flat": None, "Q_ROPE_T_TILE": 8, "Q_ROPE_H_TILE": 4, "Q_DEQUANT_WORKERS": 48, "H": 64,
        }
        exec(code, context)
        projection_m, aligned, heads, order = module.shape_facts(tokens)
        assert heads == context["dq_head_tile"], (tokens, heads, context["dq_head_tile"])
        assert not aligned or tokens % 8 == 0
        if projection_m:
            assert tokens <= 512 and projection_m == (32 if tokens < 128 else 64)
        assert order == int(tokens >= 144)
        policy = module.select_policy(tokens=tokens, requests=tokens, max_query_tokens=6, max_seq_len=131072)
        replace(policy, shape_specialization=True).validate_shape_facts(tokens)
    assert module.shape_facts(513)[0] == 0
    report = {
        "tokens_checked": 4096,
        "dq_rule_compared_to_original_source_AST": True,
        "token_boundary_points": [1, 7, 8, 16, 24, 31, 32, 127, 128, 143, 144, 511, 512, 513],
        "no_device_read": True,
        "passed": True,
    }
    (here / "SHAPE_FACTS_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
