# SPDX-License-Identifier: Apache-2.0
"""Exercise production ownership and preserve the validated computation closure."""

import ast
import json
import runpy
from pathlib import Path

import pytest
import test_output_rectangle as contracts

HERE = Path(__file__).resolve().parent
PRODUCTION = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
VALIDATED = HERE / "runs/output_rectangle_v3"


@pytest.mark.parametrize("tokens", [1, 7, 16, 64, 127, 128, 129, 143, 144, 145, 159, 160])
def test_production_column_coverage_and_wait_owners(tokens, monkeypatch):
    monkeypatch.setattr(contracts, "SOURCE", PRODUCTION)
    contracts.test_exact_columns_and_dependency_coverage(tokens)


@pytest.mark.parametrize("tokens", [1, 6, 31, 64, 96, 97, 128, 129, 144, 159, 160, 161, 168, 192, 193, 256])
def test_production_policy_boundaries(tokens, monkeypatch):
    monkeypatch.setattr(contracts, "SOURCE", PRODUCTION)
    contracts.test_only_rectangle_resource_domain_changes(tokens)


def test_original_nd_interface_is_preserved():
    functions = []
    for root in (PRODUCTION, contracts.PARENT):
        tree = ast.parse((root / "output_projection_a_stages.py").read_text())
        functions.append(
            next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "o_projection_a_stage_nd")
        )
    assert ast.dump(functions[0]) == ast.dump(functions[1])


def test_validated_device_computation_closure_is_preserved():
    class Computation(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name.endswith("_stage_nd"):
                return None
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            return self.generic_visit(node)

        def visit_Module(self, node):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                node.body = node.body[1:]
            return self.generic_visit(node)

    for path in VALIDATED.glob("*.py"):
        if path.name == "policy.py":
            continue  # Host policy values are checked independently below.
        actual, expected = [
            ast.dump(Computation().visit(ast.parse(p.read_text()))) for p in (PRODUCTION / path.name, path)
        ]
        assert actual == expected, path.name


@pytest.mark.parametrize("case_id", [f"C{i:02d}" for i in range(1, 9)])
def test_production_policy_matches_measured_descriptors(case_id):
    cases = json.loads((HERE.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    case = next(c for c in cases if c["id"] == case_id)
    descriptor = json.loads((HERE / f"descriptors/{case_id}_output_rectangle_v1.json").read_text())
    policy = runpy.run_path(str(PRODUCTION / "policy.py"))["select_policy"](
        tokens=case["token_capacity"],
        requests=case["request_capacity"],
        max_query_tokens=6,
        max_seq_len=max(case["seq_lens"]),
        compressed_table_columns=descriptor["buffers"]["compressed_table"]["shape"][1],
    )
    assert policy.constexprs() == descriptor["policy"]
