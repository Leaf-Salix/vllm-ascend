# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the independent Score replay fixture and strict diagnostics."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from audit_borrowed_score_owner import scalar_namespace
from score_owner_diagnostics import inspect_score_owner_plan

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "runs/score_owner_harness_v89"
SOURCE = HERE / "runs/borrowed_score_owner_candidate_v88"


def module_from_file(name):
    spec = importlib.util.spec_from_file_location("score_replay_" + name, HARNESS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_completed_plan(queries, histories, *, balanced=1):
    namespace = scalar_namespace(SOURCE)
    requests, tokens = 32, 168
    base = namespace["score_balance_base"](requests, tokens)
    words = [37] * (base + 96)
    descriptor = (requests + 16) // 16 * 16
    scale_base = (descriptor + 4 * tokens + 15) // 16 * 16
    words[scale_base:scale_base + 48] = [-1] * 48
    maximum = max((history + query) // 4 for query, history in zip(queries, histories))
    # All tested batches have23/28 G6 groups; the original planner's 4/8/16
    # group leaf expansion does not apply to these independent test fixtures.
    assert len(queries) in (23, 28)
    leaves = (maximum + 8191) // 8192
    plan = [len(queries), maximum, leaves, 8, 0, 6, balanced, 1]
    for group, (query, history) in enumerate(zip(queries, histories)):
        words[descriptor + 4 * group + 3] = (history + query) // 4
    namespace["score_balance_plan_stage"](words, plan, requests, tokens)
    workspace = torch.tensor(words, dtype=torch.int32)
    entry = SimpleNamespace(
        policy=SimpleNamespace(scale_page_coalescing=True, score_prefix_balance=True),
        buffers={"group_offsets": workspace, "group_plan": torch.tensor(plan, dtype=torch.int32),
                 "positions": torch.empty(tokens, dtype=torch.int64),
                 "idx_lengths": torch.empty(requests, dtype=torch.int32)},
    )
    return entry, workspace, base


@pytest.mark.parametrize("groups,balanced,identity", [(28, 1, False), (28, -1, False), (23, 1, True), (28, 0, True)])
def test_completed_plan(groups, balanced, identity):
    entry, workspace, base = make_completed_plan([6] * groups, [131072] * groups, balanced=balanced)
    result = inspect_score_owner_plan(entry, workspace, base)
    assert result["passed"] and result["identity"] is identity
    wrapper = module_from_file("run_probe").scale_writer_plan_diagnostic(entry)
    assert wrapper["score_owner_permutation"] == result


@pytest.mark.parametrize("damage", ["map", "inverse", "cost", "prefix", "stable_order", "extent", "stale_plan"])
def test_reject_damaged_plan(damage):
    entry, workspace, base = make_completed_plan([6] * 28, [131072] * 28)
    if damage == "map":
        workspace[base + 48] = 24
    elif damage == "inverse":
        workspace[base + 72] = (workspace[base + 72] + 1) % 24
    elif damage == "cost":
        workspace[base] += 1
    elif damage == "prefix":
        workspace[base + 24] += 1
    elif damage == "stable_order":
        first, second = int(workspace[base + 48]), int(workspace[base + 49])
        workspace[base + 48], workspace[base + 49] = second, first
        workspace[base + 72 + first], workspace[base + 72 + second] = 1, 0
    elif damage == "extent":
        workspace = workspace[:-1]
    elif damage == "stale_plan":
        entry.buffers["group_plan"][0] = 23
    with pytest.raises(AssertionError):
        inspect_score_owner_plan(entry, workspace, base)


@pytest.mark.parametrize("size,accepted", [(1504, True), (1505, False)])
def test_legacy_extent_remains_strict(size, accepted):
    # Bcap32/T168 long G6, original scale-layout table has1504 words.
    entry, workspace, _ = make_completed_plan([6] * 28, [131072] * 28)
    entry.policy.score_prefix_balance = False
    entry.buffers["group_offsets"] = workspace[:size]
    check = module_from_file("run_probe").scale_writer_plan_diagnostic
    if accepted:
        assert "score_owner_permutation" not in check(entry)
    else:
        with pytest.raises(AssertionError, match="Unknown"):
            check(entry)


@pytest.mark.parametrize("batch,query,capacity,requests,accepted", [
    (28, 6, 168, 32, True), (27, 6, 168, 32, False), (28, 5, 168, 32, False),
    (28, 6, 168, 28, False), (28, 6, 169, 32, False),
])
def test_submission_preflight(batch, query, capacity, requests, accepted):
    arguments = ["--metadata-matrix", "--batch", str(batch), "--query-tokens", str(query),
                 "--token-capacity", str(capacity), "--request-capacity", str(requests)]
    validate = module_from_file("submit_probe").validate_fixture_arguments
    if accepted:
        validate(arguments)
    else:
        with pytest.raises(ValueError):
            validate(arguments)


def nested_function(text, name):
    return next(node for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_strict_checks_timer_and_queued_harness_unchanged():
    original = (HERE / "metadata_matrix.py").read_text()
    revised = (HARNESS / "metadata_matrix.py").read_text()
    for name in ("load", "collect", "checked_guards", "geometry"):
        assert ast.dump(nested_function(original, name)) == ast.dump(nested_function(revised, name))
    before = nested_function(original, "candidate_run")
    after = copy.deepcopy(nested_function(revised, "candidate_run"))
    changed = 0
    for node in ast.walk(after):
        if (isinstance(node, ast.Tuple) and all(isinstance(item, ast.Constant) for item in node.elts)
                and ast.literal_eval(node) == ("signal", "ffts", "group_offsets")):
            node.elts.pop()
            changed += 1
    assert changed == 1 and ast.dump(before) == ast.dump(after)
    assert (HARNESS / "tnd_performance.py").read_bytes() == (HERE / "tnd_performance.py").read_bytes()
    manifest = json.loads((HERE / "score_owner_harness_v89.json").read_text())
    for name, expected in manifest["original_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected


def test_all_planned_cases_fit_and_reach_mapping_branches():
    tree = ast.parse((HARNESS / "metadata_matrix.py").read_text())
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate").body
    selected = []
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("cases", "steps", "history_by_id"):
                selected.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "cases.extend":
                selected.append(node)
    namespace = {"history": 131072}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "replay_case_definitions", "exec"), namespace)
    assert len(namespace["cases"]) == 9 and len(namespace["steps"]) == 4
    actual_mapping = []
    unassigned_tail_cases = 0
    for name, queries, histories, _, _ in namespace["cases"]:
        assert len(queries) == len(histories) <= 32 and all(1 <= query <= 6 for query in queries)
        assert sum(queries) <= 168 and max(histories) <= 131200
        unassigned_tail_cases += 168 - sum(queries) > (32 - len(queries)) * 6
        entry, workspace, base = make_completed_plan(queries, histories)
        result = inspect_score_owner_plan(entry, workspace, base)
        actual_mapping.append(result["identity"])
        if name == "few_groups_identity":
            assert result["identity"]
        if name in ("uniform_s6", "active_again_ascending"):
            assert not result["identity"]
    histories_by_id = namespace["history_by_id"]
    for ids, queries in namespace["steps"]:
        assert len(ids) == len(set(ids)) == len(queries) <= 32
        assert max(ids) < 32 and all(1 <= query <= 6 for query in queries) and sum(queries) <= 168
        histories = [histories_by_id[request] for request in ids]
        assert max(histories) <= 131200
        entry, workspace, base = make_completed_plan(queries, histories)
        assert inspect_score_owner_plan(entry, workspace, base)["identity"] == (len(ids) < 24)
        for request, query in zip(ids, queries):
            histories_by_id[request] += query
    assert unassigned_tail_cases > 0 and any(actual_mapping) and not all(actual_mapping)
