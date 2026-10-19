# SPDX-License-Identifier: Apache-2.0
"""Recheck layout-only measurements without enabling the candidate service."""

import ast
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_bf16_pack_20261020"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def raw():
    for name, digest in json.loads((EVIDENCE / "manifest.json").read_text()).items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
    with tarfile.open(EVIDENCE / "raw.tar.gz") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


@pytest.mark.parametrize("tokens", [21, 144, 240])
def test_all_bit_checks_and_unfiltered_paired_timing(raw, tokens):
    result = json.loads(raw[f"t{tokens}.json"])
    summary = json.loads((EVIDENCE / "summary.json").read_text())[str(tokens)]
    assert result["complete"] and result["pass"] and result["tokens"] == tokens
    assert len(result["comparisons"]) == summary["bit_comparisons"] == 536
    for value in result["comparisons"].values():
        assert value["bit_equal"] and value["different_elements"] == value["max_abs"] == 0
    assert result["helper_sha256"] == hashlib.sha256(raw["probe_bf16_input_pack.py"]).hexdigest()
    assert result["prepack_receipt_sha256"] == hashlib.sha256(raw["compile-bf16-input-pack.json"]).hexdigest()
    assert result["fixed_pack_bytes"] == 2097152
    timing = result["graph_replay_us"]
    assert not timing["outliers_removed"] and timing["workspace_lifetime_same"]
    samples = timing["samples"]
    assert all(len(values) == 12 for values in samples.values())
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    reductions = [(1 - b / a) * 100 for a, b in zip(samples["control"], samples["prepared"])]
    assert medians == timing["medians"] == summary["medians_us"]
    assert reductions == timing["paired_reduction_percent"] == summary["paired_reductions_percent"]
    assert (1 - medians["prepared"] / medians["control"]) * 100 == summary["median_reduction_percent"]
    assert sum(value > 0 for value in reductions) == summary["positive_windows"] == (8 if tokens == 21 else 12)
    assert timing["all_paired_windows_faster"] == (tokens != 21)
    receipt = json.loads(raw["compile-bf16-input-pack.json"])
    assert result["diagnostic_sources"] == {
        "prepack_candidate.py": receipt["control_sha256"],
        "bf16_input_pack_candidate.py": receipt["candidate_sha256"],
    }


def test_only_one_layout_stage_changes_and_generated_sync_matches(raw):
    receipt = json.loads(raw["compile-bf16-input-pack.json"])
    assert receipt["torch_dispatcher_registration_pass"]
    assert all(value["compiled"] and value["kernel_mode_abi_pass"] for value in receipt["programs"].values())
    for arm, name in (("control", "prepack_candidate.py"), ("candidate", "bf16_input_pack_candidate.py")):
        assert hashlib.sha256(raw[name]).hexdigest() == receipt[f"{arm}_sha256"]
        generated = raw[f"bf16-pack-{arm}.cpp"]
        assert (
            hashlib.sha256(generated).hexdigest()
            == receipt["programs"][arm]["generated"]["ptoas/precision_hc_prefix.cpp"]
        )
        assert generated.count(b"SYNCALL<SyncCoreType::Mix>();") == 20
    old = {node.name: node for node in ast.parse(raw["prepack_candidate.py"]).body if isinstance(node, ast.FunctionDef)}
    new = {
        node.name: node
        for node in ast.parse(raw["bf16_input_pack_candidate.py"]).body
        if isinstance(node, ast.FunctionDef)
    }
    assert set(new) == set(old) | {"hc_pack_input_bf16"}
    calls = [
        node
        for node in ast.walk(old["prepacked_prefix_indexer_vector"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "hc_pack_input"
    ]
    assert len(calls) == 1
    calls[0].func.id = "hc_pack_input_bf16"
    for name, node in old.items():
        assert ast.dump(node) == ast.dump(new[name])


def test_cast_move_is_exactly_reversible_to_frozen_stage(raw):
    parent = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd/hc_prefix_stages.py"
    assert parent.read_bytes() == raw["parent_hc_prefix_stages.py"]
    old = next(
        node
        for node in ast.parse(parent.read_bytes()).body
        if isinstance(node, ast.FunctionDef) and node.name == "hc_pack_input"
    )
    new = next(
        node
        for node in ast.parse(raw["bf16_input_pack_candidate.py"]).body
        if isinstance(node, ast.FunctionDef) and node.name == "hc_pack_input_bf16"
    )
    new.name = old.name
    moved = zeros = stores = 0
    for node in ast.walk(new):
        if isinstance(node, ast.For):
            for statement in list(node.body):
                if (
                    isinstance(statement, ast.Assign)
                    and ast.unparse(statement) == "panel64_values = pl.tile.cast(panel64_values, pl.BF16)"
                ):
                    node.body.remove(statement)
                    moved += 1
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.full":
            for keyword in node.keywords:
                if keyword.arg == "dtype":
                    assert ast.unparse(keyword.value) == "pl.BF16"
                    keyword.value = ast.parse("pl.FP32", mode="eval").body
                    zeros += 1
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.store":
            assert ast.unparse(node.args[0]) == "panel64_packed_values"
            node.args[0] = ast.parse("pl.tile.cast(panel64_packed_values, pl.BF16)", mode="eval").body
            stores += 1
    assert moved == zeros == stores == 1
    assert ast.dump(new) == ast.dump(old)
