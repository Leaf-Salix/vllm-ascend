# SPDX-License-Identifier: Apache-2.0
"""Validate prepack source identity, exactness and controlled-prefix evidence."""

import hashlib
import json
import statistics
import tarfile
from pathlib import Path

EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_prepack_20261020"
ROOT = Path(__file__).resolve().parents[2]


def check_exact_tree(value):
    if isinstance(value, dict):
        if "bit_equal" in value:
            assert value["bit_equal"] and value["different_elements"] == 0
        for child in value.values():
            check_exact_tree(child)
    elif isinstance(value, list):
        for child in value:
            check_exact_tree(child)


def test_final_full_root_prepack_identity_and_exactness():
    for name, digest in json.loads((EVIDENCE / "manifest.json").read_text()).items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
    receipt = json.loads((EVIDENCE / "compile-receipt.json").read_text())
    for gate in (
        "compiled",
        "kernel_mode_abi_pass",
        "torch_dispatcher_registration_pass",
        "formal_prepare_program_compiled",
        "formal_prepare_kernel_mode_abi_pass",
        "formal_operators_registered",
    ):
        assert receipt[gate]
    full = json.loads((EVIDENCE / "full-lifecycle.json").read_text())
    assert full["complete"] and full["pass"]
    assert (
        full["compile_receipt_sha256"] == hashlib.sha256((EVIDENCE / "compile-receipt.json").read_bytes()).hexdigest()
    )
    assert full["candidate_sources"] == receipt["sources"]
    assert full["parent_spmd_sources"] == receipt["parent_spmd_sources"]
    assert full["hc_packed_weight_bytes"] == 2097152
    assert full["manifest_sha256"] == hashlib.sha256((EVIDENCE / "tools/manifest.json").read_bytes()).hexdigest()
    for name, digest in full["helper_sha"].items():
        assert hashlib.sha256((EVIDENCE / "tools" / name).read_bytes()).hexdigest() == digest
    packets = full["packet_checks"]
    assert packets["readonly_after_replay"]
    assert all(packets["norm_byte_equal"].values()) and all(packets["order_byte_equal"].values())
    for package, sources in (
        ("deepseek_v4_flash_csa_tnd_precision_spmd_prepack", receipt["sources"]),
        ("deepseek_v4_flash_csa_tnd_precision_spmd", receipt["parent_spmd_sources"]),
    ):
        for name, digest in sources.items():
            path = ROOT / "vllm_ascend/ops/pypto" / package / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name, digest in receipt["baseline_sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    # Creating the second Native fixture intentionally aliases its RoPE buffer;
    # replay restores the saved inputs. Only replay comparisons are gold checks.
    for field in ("comparisons", "B_comparisons", "cross_bucket_replay_checks", "packet_checks"):
        check_exact_tree(full[field])
    check_exact_tree(full["cross_bucket_warmup"]["checks"])
    assert full["per_request_padding"] == 0 and full["physical_graph_tokens"] == 21
    assert len(full["cross_bucket_replay_checks"]) == 9
    assert full["model_loading_prepare_context"] == {
        "grad_enabled_after_constructor": True,
        "source_requires_grad": True,
        "source_inference": False,
        "packed_requires_grad": False,
    }
    previous = json.loads((EVIDENCE / "six-source-before-nograd/full-lifecycle.json").read_text())
    assert full["hc_weight_qualification"]["raw_sha256"] == previous["hc_weight_qualification"]["raw_sha256"]


def test_both_prefix_shapes_preserve_every_comparison():
    with tarfile.open(EVIDENCE / "prefix-raw.tar.gz") as archive:
        for tokens in (21, 240):
            result = json.load(archive.extractfile(f"results/prepack-ab-t{tokens}-v1/result.json"))
            assert result["complete"] and result["pass"] and result["tokens"] == tokens
            assert len(result["comparisons"]) == 491
            check_exact_tree(result["comparisons"])
            assert result["fixed_pack_bytes"] == 2097152
            timing = result["graph_replay_us"]
            old, new = (timing["samples"][arm] for arm in ("control", "prepared"))
            assert len(old) == len(new) == 12 and all(a > b for a, b in zip(old, new))
            assert not timing["outliers_removed"] and timing["workspace_lifetime_same"]
            for arm in ("control", "prepared"):
                assert statistics.median(timing["samples"][arm]) == timing["medians"][arm]
            ratio = (1 - statistics.median(new) / statistics.median(old)) * 100
            assert ratio == timing["median_reduction_percent"]
