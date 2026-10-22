# SPDX-License-Identifier: Apache-2.0
"""Check actual joint-grid ownership and compatibility with existing consumers."""

import ast
import collections
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/joint_prefix_v2"
PRODUCTION = HERE.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
WIDTHS = {"idx_value": 256, "idx_score": 256, "main_value": 1024, "main_score": 1024, "kv_fp32": 512}


def grid(tokens, row_tile):
    tree = ast.parse((SOURCE / "joint_prefix_stages.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "joint_bf16_prefix_stage")
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None
    records = []
    namespace = dict(
        pl=SimpleNamespace(range=range, min=min, tensor=SimpleNamespace(dim=lambda x, _: x)),
        JOBS_PER_ROW_BLOCK=24,
        N_TILE=128,
    )
    mapping = {
        "idx_wkv": "idx_value",
        "idx_wgate": "idx_score",
        "main_wkv": "main_value",
        "main_wgate": "main_score",
        "wkv": "kv_fp32",
    }

    def emit(x, weight, output, row, column, rows, tile, transpose):
        assert x == tokens and tile == row_tile and mapping[weight] == output
        assert transpose == (weight != "wkv")
        records.append((namespace["owner"], output, row, rows, column))
        return output

    namespace["_project_indexer"] = lambda *args: emit(*args, True)
    namespace["_project_main"] = lambda *args: emit(*args, True)
    namespace["_project_forward"] = lambda *args: emit(*args, False)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "actual_joint_grid", "exec"),
        namespace,
    )
    for worker in range(24):
        namespace["owner"] = worker
        result = namespace[node.name](
            tokens,
            "idx_wkv",
            "idx_wgate",
            "idx_value",
            "idx_score",
            "main_wkv",
            "main_wgate",
            "main_value",
            "main_score",
            "wkv",
            "kv_fp32",
            worker,
            24,
            row_tile,
        )
        assert result == tuple(WIDTHS)
    return records


@pytest.mark.parametrize(
    "tokens", [0, 1, 6, 32, 64, 65, 80, 95, 96, 127, 128, 129, 143, 144, 167, 168, 191, 192, 193, 256]
)
@pytest.mark.parametrize("row_tile", [128, 192])
def test_each_real_output_rectangle_has_exactly_one_writer(tokens, row_tile):
    records = grid(tokens, row_tile)
    actual = collections.Counter(
        (output, row, column // 128) for _, output, first, rows, column in records for row in range(first, first + rows)
    )
    expected = collections.Counter(
        {
            (name, row, column): 1
            for name, width in WIDTHS.items()
            for row in range(tokens)
            for column in range(width // 128)
        }
    )
    assert actual == expected
    jobs = collections.Counter(owner for owner, *_ in records)
    assert [jobs[owner] for owner in range(24)] == [math.ceil(tokens / row_tile)] * 24


@pytest.mark.parametrize("tokens", [65, 80, 96, 127, 128, 129, 144, 168, 192])
def test_existing_consumers_acquire_every_new_writer(tokens):
    records = grid(tokens, 128 if tokens <= 128 else 192)
    tree = ast.parse((SOURCE / "vector_prefix_stages.py").read_text())
    expressions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("idx_tasks", "main_tasks"):
                expressions[node.targets[0].id] = compile(ast.Expression(node.value), "actual_consumer_count", "eval")
    for prefix, outputs in [("idx", ("idx_value", "idx_score")), ("main", ("main_value", "main_score"))]:
        count = eval(expressions[prefix + "_tasks"], dict(tokens=tokens))
        acquired = {ordinal % 24 for ordinal in range(min(count, 24))}
        actual = {owner for owner, output, *_ in records if output in outputs}
        assert actual <= acquired
    kv = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "original_kv_vector_stage")
    cooperative = next(
        n
        for n in ast.walk(kv)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "COOPERATIVE_PREFIX"
    )
    wait = next(
        n
        for block in cooperative.body
        for n in ast.walk(block)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "wait_row_owners"
    )
    assert [ast.unparse(wait.args[i]) for i in (1, 4, 5, 6, 7, 8, 9)] == [
        "PHASE_KV_PROJECT",
        "0",
        "GROUPS",
        "1",
        "GROUPS",
        "0",
        "1",
    ]
    assert {owner for owner, output, *_ in records if output == "kv_fp32"} <= set(range(24))


@pytest.mark.parametrize("tokens", [1, 6, 16, 32, 48, 56, 64, 65, 96, 128, 129, 144, 168, 192, 193, 256])
def test_policy_changes_only_joint_projection_controls(tokens):
    old = runpy.run_path(str(PRODUCTION / "policy.py"))["select_policy"]
    new = runpy.run_path(str(SOURCE / "policy.py"))["select_policy"]
    for width in [1, 5, 6]:
        for history in [8192, 131072, 262144]:
            arguments = dict(
                tokens=tokens, requests=math.ceil(tokens / width), max_query_tokens=width, max_seq_len=history + width
            )
            actual = new(**arguments).constexprs()
            assert actual.pop("JOINT_BF16_PREFIX") == (64 < tokens <= 192)
            assert actual.pop("PREFIX_PROJECTION_ROWS") == (192 if 128 < tokens <= 192 else 128)
            assert actual == old(**arguments).constexprs()


def test_publication_order_and_no_duplicate_main_or_kv():
    tree = ast.parse((SOURCE / "cube_schedule.py").read_text())
    joint = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "JOINT_BF16_PREFIX and COOPERATIVE_PREFIX"
    )
    phases = [
        n.value.args[1].id
        for n in joint.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "publish_phase"
    ]
    assert phases == ["PHASE_IDX_PROJECT", "PHASE_MAIN_PROJECT", "PHASE_KV_PROJECT"]
    qa = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "q_a_stage"
    )
    index_query = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "indexer_query_project_stage"
    )
    assert qa.lineno < joint.lineno < index_query.lineno
    overlap = ast.parse((SOURCE / "projection_overlap_stages.py").read_text())
    off = [n for n in ast.walk(overlap) if isinstance(n, ast.If) and ast.unparse(n.test) == "not JOINT_BF16_PREFIX"]
    assert len(off) == 2
    guarded = {
        n.func.id for block in off for n in ast.walk(block) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert {"compressor_project_stage", "kv_project_stage"} <= guarded


def test_native_bindings_and_other_math_are_preserved():
    for name in [
        "binding.py",
        "qkv_stages.py",
        "indexer_query_stages.py",
        "indexer_score_stages.py",
        "indexer_publish_stages.py",
        "attention_plan_stages.py",
        "attention_stream_stages.py",
        "vector_prefix_stages.py",
        "vector_schedule.py",
        "output_projection_a_stages.py",
        "output_projection_b_stages.py",
        "phase_sync.py",
        "signals.py",
    ]:
        assert (SOURCE / name).read_bytes() == (PRODUCTION / name).read_bytes()
