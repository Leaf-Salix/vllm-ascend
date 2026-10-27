# SPDX-License-Identifier: Apache-2.0
"""不用NPU或编译器验证Native HC容量、真实TND和保留的外部ABI。"""

import ast
import importlib
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
package = types.ModuleType("hc_contract_only")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
policy = importlib.import_module("hc_contract_only.policy")
config = importlib.import_module("hc_contract_only.service_config")


def test_ragged_capacity_does_not_require_six_multiple():
    assert config.supported_descriptor(
        tokens=17, requests=4, actual_tokens=17, max_query_tokens=6, max_seq_len=8198, batch_capacity=40
    )
    selected = policy.select_policy(tokens=17, requests=4, max_query_tokens=6, max_seq_len=8198)
    assert selected.constexprs()["HC_CONSUMER_REDUCE"] is False


def test_unsupported_hc_capacity_falls_back_before_kernel():
    assert not config.supported_descriptor(
        tokens=258, requests=43, actual_tokens=258, max_query_tokens=6, max_seq_len=8198, batch_capacity=64
    )
    with pytest.raises(NotImplementedError, match="256"):
        policy.select_policy(tokens=258, requests=43, max_query_tokens=6, max_seq_len=8198)


def test_old_four_split_consumer_reduce_is_rejected():
    selected = policy.select_policy(tokens=24, requests=4, max_query_tokens=6, max_seq_len=8198)
    with pytest.raises(ValueError, match="consumer_reduce"):
        replace(selected, hc_consumer_reduce=True).constexprs()


@pytest.mark.parametrize(("name", "tensors"), [("decode_csa_spmd", 55), ("decode_csa_spmd_debug", 57)])
def test_public_root_abi_is_preserved(name, tensors):
    tree = ast.parse((SOURCE / "entry.py").read_text())
    root = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assert sum(ast.unparse(arg.annotation) != "pl.constexpr" for arg in root.args.args) == tensors


def test_original_cube_weight_cache_policy_indices_are_preserved():
    tree = ast.parse((SOURCE / "cube_schedule.py").read_text())
    root = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cube_schedule")
    args = [arg.arg for arg in root.args.args]
    assert {index: args[index] for index in (1, 8, 10, 50, 53)} == {
        1: "wq_a",
        8: "wq_b",
        10: "wkv",
        50: "wo_a",
        53: "wo_b",
    }


def test_observation_calls_the_unmodified_production_core():
    production = ast.parse((SOURCE / "entry.py").read_text())
    root = next(
        node for node in production.body if isinstance(node, ast.FunctionDef) and node.name == "decode_csa_spmd_debug"
    )
    observed = ast.parse((Path(__file__).parent / "spmd_probe_entry.py").read_text())

    def core_call(tree):
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_csa_core_group"
        )

    assert ast.dump(core_call(root)) == ast.dump(core_call(observed))
