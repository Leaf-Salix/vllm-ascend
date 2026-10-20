# SPDX-License-Identifier: Apache-2.0
"""Audit the known failed acceptance; passing this audit is not a precision pass."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_bf16layout_first_difference_20261020"
ROOT = Path(__file__).resolve().parents[2]


def read(name):
    path = EVIDENCE / name
    return json.loads(path.read_bytes() if path.is_file() else helper_bytes(name))


def helper_bytes(name):
    with tarfile.open(EVIDENCE / "raw-helpers.tar.gz") as archive:
        return archive.extractfile(name).read()


def exact_tree(value):
    if isinstance(value, dict):
        if "bit_equal" in value:
            assert value["bit_equal"] and value["different_elements"] == 0
        for child in value.values():
            exact_tree(child)
    elif isinstance(value, list):
        for child in value:
            exact_tree(child)


def test_archive_sources_and_stopped_status():
    for name, digest in read("manifest.json").items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
    status = read("status.json")
    assert not status["accepted"] and status["first_difference_stops_expansion"]
    assert status["T21_pass"] and not status["T126_pass"]
    assert not any(status[key] for key in ("T240_run", "whole_8k_run", "whole_128k_run"))
    receipt = read("compile-receipt.json")
    assert len(receipt["sources"]) == 7 and receipt["formal_operators_registered"]
    assert receipt["kernel_mode_abi_pass"] and receipt["formal_prepare_kernel_mode_abi_pass"]
    for name, digest in receipt["sources"].items():
        path = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd_bf16layout" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    generated = helper_bytes("generated-prefix.cpp")
    assert hashlib.sha256(generated).hexdigest() == read("generated-audit.json")["source_sha256"]
    assert generated.count(b"SYNCALL<SyncCoreType::Mix>();") == 20


def test_short_full_pass_and_both_large_failures_bind_actual_helpers():
    short = read("t21-pass.json")
    assert short["pass"] and short["complete"] and short["true_tokens"] == 21
    assert len(short["cross_bucket_replay_checks"]) == 9
    for field in ("comparisons", "B_comparisons", "cross_bucket_replay_checks"):
        exact_tree(short[field])
    exact_tree(short["cross_bucket_warmup"]["checks"])
    for filename, tools, receipt_name in (
        ("t21-pass.json", "bf16layout-full-tools", "compile-receipt.json"),
        ("t126-failure.json", "bf16layout-full-tools", "compile-receipt.json"),
        ("t126-control-failure.json", "bf16layout-control-full-tools", "parent-compile-receipt.json"),
    ):
        data = read(filename)
        assert data["candidate_sources"] == read(receipt_name)["sources"]
        assert data["compile_receipt_sha256"] == hashlib.sha256(helper_bytes(receipt_name)).hexdigest()
        for name, digest in data["helper_sha"].items():
            assert hashlib.sha256(helper_bytes(f"{tools}/{name}")).hexdigest() == digest
        if filename != "t21-pass.json":
            assert not data["pass"] and not data["complete"] and data["true_tokens"] == 126
            assert data["physical_graph_tokens"] == 126 and data["per_request_padding"] == 0
            assert data["failed_boundaries"] == ["x_out", "idx_topk"]
            metrics = data["comparisons"]["candidate"]
            assert metrics["idx_topk"]["different_elements"] == 28
            assert metrics["x_out"]["different_elements"] == 1164
            assert metrics["x_out"]["max_abs"] == 0.015625
            for name, metric in metrics.items():
                if name not in ("x_out", "idx_topk"):
                    assert metric["bit_equal"]


def test_saved_arrays_recompute_reordering_and_identical_original_control():
    with np.load(EVIDENCE / "minimal-difference.npz", allow_pickle=False) as data:
        for arm in ("native", "candidate"):
            for kind in ("topk", "output_changed_rows_bits"):
                assert np.array_equal(data[f"bf16_{arm}_{kind}"], data[f"control_{arm}_{kind}"])
        native, candidate = data["bf16_native_topk"], data["bf16_candidate_topk"]
        assert native.shape == candidate.shape == (126, 512)
        assert np.count_nonzero(native != candidate) == 28
        assert np.flatnonzero(np.any(native != candidate, axis=1)).tolist() == [86]
        assert np.array_equal(np.sort(native, axis=1), np.sort(candidate, axis=1))
        assert data["bf16_output_rows"].tolist() == data["control_output_rows"].tolist() == [86, 103]
        left, right = data["bf16_native_output_changed_rows_bits"], data["bf16_candidate_output_changed_rows_bits"]
        assert np.count_nonzero(left != right) == 1164
        left_float = (left.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        right_float = (right.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        assert float(np.max(np.abs(left_float - right_float))) == 0.015625
    comparison = read("snapshot-comparison.json")
    for field in ("states", "inputs", "initial_allocations"):
        assert all(all(values.values()) for values in comparison[field].values())
    assert comparison["tensor_sha256"]["bf16"] == comparison["tensor_sha256"]["control"]
    assert comparison["bf16_output_difference"] == comparison["control_output_difference"]
    assert comparison["bf16_output_difference"]["rows"] == [86, 103]


def test_unsupported_score_export_has_no_precision_claim():
    data = read("score-export-unavailable.json")
    assert not data["pass"] and not data["complete"]
    assert "returnValues only supported False" in data["error"]
    assert not data.get("score_probe_exported", False)
