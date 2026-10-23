# SPDX-License-Identifier: Apache-2.0
"""Run existing service contracts against the isolated tier and check selection."""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOADER_SPEC = importlib.util.spec_from_file_location(
    "capacity_tier_source_loader",
    ROOT / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/source_loader.py",
)
LOADER = importlib.util.module_from_spec(LOADER_SPEC)
LOADER_SPEC.loader.exec_module(LOADER)
load_package = LOADER.load_package
SOURCE = HERE / "runs/physical_capacity_tier_v1"
PACKAGE = load_package(SOURCE, "capacity_tier_cpu")
LEGACY = load_package(HERE / "runs/merged_schedule_v1", "capacity_tier_legacy_cpu")
select_policy = importlib.import_module(PACKAGE.name + ".policy").select_policy
legacy_select = importlib.import_module(LEGACY.name + ".policy").select_policy

# Preserve the original test bodies, replacing only their operator import root.
contracts = ROOT / "tests/ut/ops/test_dsv4_csa_spmd_tnd.py"
tree = ast.parse(contracts.read_text())
prefix = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd"
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
        node.module = PACKAGE.name + node.module[len(prefix) :]
exec(compile(tree, str(contracts), "exec"), globals())


@pytest.mark.parametrize("tokens", [6, 56, 96])
@pytest.mark.parametrize("columns", [None, 1, 2048, 2049, 2050, 2080, 2081, 4096])
def test_capacity_tier_uses_allocated_columns_without_changing_other_policies(tokens, columns):
    kwargs = dict(tokens=tokens, requests=16, max_query_tokens=6, max_seq_len=262150)
    selected = select_policy(**kwargs, compressed_table_columns=columns)
    expected = 2080 if tokens >= 64 and columns is not None and 2048 < columns <= 2080 else 2048
    assert selected.plan_table_columns == expected
    values = selected.constexprs()
    assert values.pop("PLAN_TABLE_COLUMNS") == expected
    assert values == legacy_select(**kwargs).constexprs()


@pytest.mark.parametrize("columns", [True, 0, -1, 2050.0])
def test_capacity_tier_rejects_invalid_shape_descriptors(columns):
    with pytest.raises(ValueError, match="compressed_table_columns"):
        select_policy(tokens=96, requests=16, max_query_tokens=6, max_seq_len=262150, compressed_table_columns=columns)


def test_capacity_is_forwarded_to_every_producer_consumer_and_service_selector():
    expected = {
        "attention_plan_stages.py": {"physical_plan_supported": 2},
        "attention_stream_stages.py": {"physical_plan_supported": 1},
        "vector_schedule.py": {
            "attention_geometry_prepare_stage": 1,
            "attention_slot_plan_stage": 1,
            "attention_qk_pv_vector_lane_stage": 1,
        },
        "entry.py": {"vector_schedule": 1, "_csa_core_group": 1},
    }
    for filename, functions in expected.items():
        parsed = ast.parse((SOURCE / filename).read_text())
        for name, count in functions.items():
            calls = [
                node
                for node in ast.walk(parsed)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            ]
            assert len(calls) == count
            assert all(
                any(
                    kw.arg == "PLAN_TABLE_COLUMNS" and ast.unparse(kw.value) == "PLAN_TABLE_COLUMNS"
                    for kw in call.keywords
                )
                for call in calls
            )
    for filename, target, count in (("binding.py", "select_policy", 1), ("service.py", "self._select_policy", 2)):
        calls = [
            node
            for node in ast.walk(ast.parse((SOURCE / filename).read_text()))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == target
        ]
        assert len(calls) == count
        assert all(any(kw.arg == "compressed_table_columns" for kw in call.keywords) for call in calls)
