# SPDX-License-Identifier: Apache-2.0
"""CPU tests for complete-page evidence and the isolated WS replay fixture."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scale_layout_diagnostics import inspect_scale_layout

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "runs/ws_layout_harness_v90"

if not HARNESS.is_dir():
    pytest.skip("Requires the generated local v90 WS replay harness", allow_module_level=True)


def module_from_file(name):
    spec = importlib.util.spec_from_file_location("ws_replay_" + name, HARNESS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry_for_pages(ids, length, proof, *, physical_pages=2048):
    requests, tokens = 4, 16
    base = 192
    table = torch.zeros((requests, len(ids)), dtype=torch.int32)
    table[0] = torch.tensor(ids, dtype=torch.int32)
    words = torch.full((base + 2 * requests,), 37, dtype=torch.int32)
    words[80:128] = -1
    words[base:] = 0
    words[base:base + 2] = torch.tensor(proof, dtype=torch.int32)
    entry = SimpleNamespace(
        policy=SimpleNamespace(scale_page_coalescing=True, balanced=True, score_ws_pool_panels=8),
        buffers={"idx_bounds": torch.tensor([0, 6, 6, 6, 6], dtype=torch.int32),
                 "idx_lengths": torch.tensor([length, 0, 0, 0], dtype=torch.int32),
                 "positions": torch.empty(tokens, dtype=torch.int64), "indexer_table": table,
                 "group_offsets": words, "idx_cache": SimpleNamespace(shape=(physical_pages, 4160))},
    )
    return entry, words, base


@pytest.mark.parametrize("ids,length,proof,physical_pages", [
    (list(range(10, 18)), 1024, (10, 1), 2048),
    (list(range(17, 9, -1)), 1024, (17, -1), 2048),
    ([42], 128, (42, 1), 2048),
    ([10, 12, 11, 13, 14, 15, 16, 17], 1024, (0, 0), 2048),
    (list(range(10, 18)) + [99, 2], 1024, (10, 1), 2048),
    ([10, 11, -1, 13, 14, 15, 16, 17], 1024, (0, 0), 2048),
    (list(range(10, 18)), 1152, (0, 0), 2048),
    (list(range(10, 18)), 1024, (0, 0), (1 << 24) + 1),
    (list(range((1 << 24) - 8, 1 << 24)), 1024, ((1 << 24) - 8, 1), 1 << 24),
    (list(range(10, 18)), 0, (0, 0), 2048),
    (list(range(10, 18)), 4, (0, 0), 2048),
])
def test_valid_layout_or_fallback(ids, length, proof, physical_pages):
    entry, workspace, base = entry_for_pages(ids, length, proof, physical_pages=physical_pages)
    result = inspect_scale_layout(entry, workspace, base)
    assert result["passed"] and result["requests"][0]["direction"] == proof[1]
    assert module_from_file("run_probe").scale_writer_plan_diagnostic(entry)["scale_page_layout"] == result


@pytest.mark.parametrize("damage", ["first", "direction", "false_fallback", "interior", "inactive", "extent"])
def test_reject_incorrect_or_stale_proof(damage):
    entry, workspace, base = entry_for_pages(list(range(10, 18)), 1024, (10, 1))
    if damage == "first":
        workspace[base] += 1
    elif damage == "direction":
        workspace[base + 1] = -1
    elif damage == "false_fallback":
        workspace[base:base + 2] = 0
    elif damage == "interior":
        entry.buffers["indexer_table"][0, 3] = 12
    elif damage == "inactive":
        entry.buffers["idx_lengths"][0] = 0
    elif damage == "extent":
        workspace = workspace[:-1]
    with pytest.raises(AssertionError):
        inspect_scale_layout(entry, workspace, base)


def test_out_of_capacity_request_is_inactive():
    entry, workspace, base = entry_for_pages(list(range(10, 18)), 1024, (0, 0))
    entry.buffers["idx_bounds"][1] = 17
    assert inspect_scale_layout(entry, workspace, base)["inactive"] == [0, 1, 2, 3]


@pytest.mark.parametrize("batch,query,capacity,requests,accepted", [
    (6, 6, 65, 16, True), (6, 5, 65, 16, False), (6, 6, 65, 12, False),
    (6, 6, 63, 16, False), (5, 6, 65, 16, False),
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


def function_node(text, name):
    return next(node for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_existing_precision_protocol_and_timer_remain():
    old = (HERE / "metadata_matrix.py").read_text()
    new = (HARNESS / "metadata_matrix.py").read_text()
    for name in ("collect", "checked_guards", "load", "geometry"):
        assert ast.dump(function_node(old, name)) == ast.dump(function_node(new, name))
    revised = copy.deepcopy(function_node(new, "candidate_run"))
    changes = 0
    for node in ast.walk(revised):
        if (isinstance(node, ast.Tuple) and all(isinstance(item, ast.Constant) for item in node.elts)
                and ast.literal_eval(node) == ("signal", "ffts", "group_offsets")):
            node.elts.pop()
            changes += 1
    assert changes == 1 and ast.dump(revised) == ast.dump(function_node(old, "candidate_run"))
    assert (HARNESS / "tnd_performance.py").read_bytes() == (HERE / "tnd_performance.py").read_bytes()
    manifest = json.loads((HERE / "ws_layout_harness_v90.json").read_text())
    for name, expected in manifest["original_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected


def test_all_cases_fit_and_required_page_directions_are_real():
    tree = ast.parse((HARNESS / "metadata_matrix.py").read_text())
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate").body
    selected = []
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("cases", "steps", "history_by_id", "expected_directions"):
                selected.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "cases.extend":
                selected.append(node)
    namespace = {"history": 131072}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "ws_replay_cases", "exec"), namespace)
    assert len(namespace["cases"]) == 12 and len(namespace["steps"]) == 4
    columns = (131200 + 6 + 127) // 128 + 1
    for name, queries, histories, _, pattern in namespace["cases"]:
        assert len(queries) == len(histories) <= 16 and all(1 <= query <= 6 for query in queries)
        assert sum(queries) <= 65 and 65 - sum(queries) <= (16 - len(queries)) * 6
        for query, history in zip(queries, histories):
            assert history <= 131200
            pages = list(range(columns, 0, -1))
            full_history_pages = history // 128
            if pattern == "ascending_history":
                pages[:full_history_pages] = reversed(pages[:full_history_pages])
            elif pattern == "fragmented_history":
                pages[1], pages[2] = pages[2], pages[1]
            if name in namespace["expected_directions"]:
                direction = namespace["expected_directions"][name]
                proof = (pages[0], direction) if direction else (0, 0)
                entry, workspace, base = entry_for_pages(pages, history + query, proof)
                inspect_scale_layout(entry, workspace, base)
    for ids, queries in namespace["steps"]:
        assert len(ids) == len(set(ids)) == len(queries) <= 16
        assert max(ids) < 16 and all(1 <= query <= 6 for query in queries)
        assert sum(queries) <= 65 and 65 - sum(queries) <= (16 - len(ids)) * 6
