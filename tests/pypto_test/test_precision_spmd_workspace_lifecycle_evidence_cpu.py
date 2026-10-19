# SPDX-License-Identifier: Apache-2.0
"""Keep initial failures and bind multi-bucket FULL replay to current sources."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd"
EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_workspace_lifecycle_20261020"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(name):
    return json.loads(gzip.decompress((EVIDENCE / name).read_bytes()))


def archive():
    with tarfile.open(fileobj=io.BytesIO((EVIDENCE / "tools.tar.gz").read_bytes()), mode="r:gz") as contents:
        return {m.name: contents.extractfile(m).read() for m in contents.getmembers()}


def test_current_source_identity_and_preserved_initialization_failure():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert len(manifest) == 5
    assert all(sha((EVIDENCE / name).read_bytes()) == digest for name, digest in manifest.items())
    compiled = read("compile-lifecycle.json.gz")
    assert len(compiled["sources"]) == 14 and len(compiled["baseline_sources"]) == 43
    assert all(sha((PACKAGE / name).read_bytes()) == digest for name, digest in compiled["sources"].items())
    assert all(sha((ROOT / name).read_bytes()) == digest for name, digest in compiled["baseline_sources"].items())
    assert compiled["compiled"] and compiled["kernel_mode_abi_pass"]
    assert compiled["torch_dispatcher_registration_pass"]
    assert compiled["torch_schema"].split(" -> ")[0].count("Tensor") == 64
    result = read("lifecycle-pass.json.gz")
    assert result["candidate_sources"] == compiled["sources"]
    assert result["compile_receipt_sha256"] == sha(
        gzip.decompress((EVIDENCE / "compile-lifecycle.json.gz").read_bytes())
    )
    before = json.loads(
        gzip.decompress((EVIDENCE.parent / "spmd_full_root_20261020/compile-static.json.gz").read_bytes())
    )
    assert {name for name in compiled["sources"] if compiled["sources"][name] != before["sources"][name]} == {
        "workspace.py"
    }
    assert sha(archive()["workspace-before-lifecycle.py"]) == before["sources"]["workspace.py"]
    failed = read("failed-whole-initialization.json.gz")
    assert set(failed["ranks"]) == {str(i) for i in range(16)}
    for row in failed["ranks"].values():
        assert row["stream_guard_error"]
        assert not row["token_stop_exists"] and not row["measured_result_exists"]


def test_multi_bucket_full_replay_is_exact_with_original_readonly_inputs():
    result = read("lifecycle-pass.json.gz")
    assert result["complete"] and result["pass"]
    assert result["lengths"] == [1, 2, 3, 4, 5, 6]
    assert result["true_tokens"] == result["physical_graph_tokens"] == 21
    assert result["per_request_padding"] == 0
    tools = archive()
    for name, digest in result["helper_sha"].items():
        assert sha(tools[f"bucket-lifecycle-tools-v2/{name}"]) == digest
    assert sha(tools["bucket-lifecycle-tools-v2/manifest.json"]) == result["manifest_sha256"]
    for checks in (result["comparisons"]["candidate"], result["B_comparisons"]["candidate"]):
        assert len(checks) == 8 and all(m["bit_equal"] for m in checks.values())
    warmup = result["cross_bucket_warmup"]
    assert warmup["first_tokens"] == 21 and warmup["second_tokens"] == 6
    assert warmup["different_stream"] and warmup["second_actual_FULL_graph"]
    assert warmup["capture_stream"] != warmup["compute_stream"]
    assert warmup["capture_stream"] == result["workspace_capture_stream"]
    assert warmup["checks_count"] == 80 and warmup["concurrent_replay_tested"] is False
    assert len(warmup["checks"]) == 8 and all(m["bit_equal"] for m in warmup["checks"].values())
    checks = result["cross_bucket_replay_checks"]
    assert set(checks) == {f"{i}/{label}" for i in range(3) for label in ("21a", "6", "21b")}
    assert all(len(row) == 8 and all(m["bit_equal"] for m in row.values()) for row in checks.values())
    for graph in result["graphs"].values():
        assert graph["actual_acl_graph"] and graph["fullgraph"] and graph["live_input_ABA"]
        assert all(m["status"] == "PASS" for m in graph["guards"].values())
    assert result["packet_checks"]["readonly_after_replay"]


def test_input_alias_failure_remains_visible_and_restoration_uses_saved_gold():
    failed, passed = read("failed-lifecycle-inputs.json.gz"), read("lifecycle-pass.json.gz")
    assert not failed["pass"]
    assert all(m["bit_equal"] for m in failed["cross_bucket_warmup"]["checks"].values())
    assert any(not m["bit_equal"] for m in failed["cross_bucket_replay_checks"]["0/21a"].values())
    changes = passed["readonly_after_second_fixture"]
    assert {name for name, m in changes.items() if not m["bit_equal"]} == {"rope.cos", "rope.sin"}
    assert changes["rope.cos"]["different_elements"] == 314
    assert changes["rope.sin"]["different_elements"] == 320
    for row in passed["shared_rope_views"].values():
        assert row["first_ptr"] == row["second_ptr"] and row["same_storage"]
        assert row["first_shape"] == [21, 1, 1, 64] and row["second_shape"] == [6, 1, 1, 64]
    helper = archive()["bucket-lifecycle-tools-v2/probe_csa.py"].decode()
    assert "for live,expected in fixture['readonly'].values():live.copy_(expected)" in helper
    assert "restore(fixture);restore_readonly(fixture)" in helper
    assert "metadata input differs before replay" in helper
