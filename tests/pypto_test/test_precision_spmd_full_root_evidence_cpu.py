# SPDX-License-Identifier: Apache-2.0
"""Bind full-root passes and the static T6 first-difference chain to exact sources."""

import ast
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd"
EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_full_root_20261020"


def read(name):
    return json.loads(gzip.decompress((EVIDENCE / name).read_bytes()))


def archive(name):
    with tarfile.open(fileobj=io.BytesIO((EVIDENCE / name).read_bytes()), mode="r:gz") as contents:
        return {m.name: contents.extractfile(m).read() for m in contents.getmembers()}


def witness(name):
    return torch.load(
        io.BytesIO(gzip.decompress((EVIDENCE / name).read_bytes())), map_location="cpu", weights_only=True
    )


def sha(value):
    return hashlib.sha256(value).hexdigest()


def test_archived_identity_preserves_old_sources_and_qualifies_current_root():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert len(manifest) == 16
    assert all(sha((EVIDENCE / n).read_bytes()) == h for n, h in manifest.items())
    old, new = read("compile-before-static.json.gz"), read("compile-static.json.gz")
    assert len(new["sources"]) == len(old["sources"]) == 14
    # These runs preceded the separately qualified cross-bucket warmup fence.
    # Keep their historical workspace identity; the lifecycle evidence binds all
    # fourteen current sources instead of relabelling these old hardware runs.
    lifecycle = EVIDENCE.parent / "spmd_workspace_lifecycle_20261020"
    with tarfile.open(lifecycle / "tools.tar.gz", mode="r:gz") as tools:
        historical_workspace = tools.extractfile("workspace-before-lifecycle.py").read()
    for name, digest in new["sources"].items():
        value = historical_workspace if name == "workspace.py" else (PACKAGE / name).read_bytes()
        assert sha(value) == digest
    assert {n for n in old["sources"] if old["sources"][n] != new["sources"][n]} == {"service.py"}
    assert sha(archive("tools.tar.gz")["service-before-static.py"]) == old["sources"]["service.py"]
    assert old["baseline_sources"] == new["baseline_sources"] and len(new["baseline_sources"]) == 43
    for name, digest in new["baseline_sources"].items():
        assert sha((ROOT / name).read_bytes()) == digest
    for gate in ("compiled", "kernel_mode_abi_pass", "torch_dispatcher_registration_pass"):
        assert old[gate] and new[gate]
    assert new["torch_schema"].split(" -> ")[0].count("Tensor") == 64


@pytest.mark.parametrize(
    ("name", "lengths", "fixed"),
    [
        ("short4-before-static.json.gz", [1, 1, 1, 1], False),
        ("mixed21-before-static.json.gz", [1, 2, 3, 4, 5, 6], False),
        ("t6-static-fixed.json.gz", [6], True),
    ],
)
def test_full_graph_native_boundaries_and_actual_padding(name, lengths, fixed):
    result = read(name)
    assert result["complete"] and result["pass"]
    assert result["lengths"] == lengths
    assert result["true_tokens"] == result["physical_graph_tokens"] == sum(lengths)
    assert result["per_request_padding"] == 0
    compiled = read("compile-static.json.gz" if fixed else "compile-before-static.json.gz")
    assert result["candidate_sources"] == compiled["sources"]
    assert result["compile_receipt_sha256"] == sha(
        gzip.decompress(
            (EVIDENCE / ("compile-static.json.gz" if fixed else "compile-before-static.json.gz")).read_bytes()
        )
    )
    for checks in (result["comparisons"]["candidate"], result["B_comparisons"]["candidate"]):
        assert len(checks) == 8
        assert all(m["bit_equal"] and m["different_elements"] == 0 for m in checks.values())
    for graph in result["graphs"].values():
        assert graph["actual_acl_graph"] and graph["fullgraph"] and graph["compiled_raw_equal"]
        assert graph["live_input_ABA"] and graph["topk_poison_replay_restored"]
        assert graph["options"]["force_eager"] is False and graph["native_super_kernel_optimize"] is False
        assert all(m["status"] == "PASS" for m in graph["guards"].values())
    if fixed:
        assert result["actual_order_packet"]["key"] == [1, 1, 0]
    if len(lengths) == 6 or fixed:
        checks = result["two_root_replay_checks"]
        assert set(checks) == {"0", "1", "2"}
        assert all(len(repeat) == 16 and all(m["bit_equal"] for m in repeat.values()) for repeat in checks.values())
    tools = archive("tools.tar.gz")
    folders = ["full-root-static-tools"] if fixed else ["full-root-tools", "full-root-pool-tools"]
    assert any(all(sha(tools[f"{folder}/{n}"]) == h for n, h in result["helper_sha"].items()) for folder in folders)


def test_t6_first_difference_is_not_hidden_by_successful_diagnostic_exit():
    failed = read("failed-t6.json.gz")
    assert not failed["pass"] and failed["failed_boundaries"] == ["idx_topk"]
    assert failed["comparisons"]["candidate"]["idx_topk"]["different_elements"] == 155
    triage = read("t6-triage.json.gz")
    assert triage["complete"] and triage["pass"] is False and triage["timing_executed"] is False
    assert all(triage["saved_A_input_exact"].values())
    assert all(triage["previous_native_FULL_exact"].values())
    assert all(triage["previous_candidate_FULL_exact"].values())
    assert all(m["bit_equal"] for m in triage["candidate_vs_control"].values())
    for arm in ("control", "candidate"):
        checks = triage["comparisons"][arm]
        assert {k for k, m in checks.items() if not m["bit_equal"]} == {"idx_topk"}
        assert checks["idx_topk"]["different_elements"] == 155
    data = witness("qli-witness.pt.gz")
    native = data["states"]["native"]["idx_topk"].reshape(6, 512)
    pto = data["states"]["candidate"]["idx_topk"].reshape(6, 512)
    assert torch.equal(native.sort().values, pto.sort().values)
    assert torch.equal(native[:2], pto[:2]) and torch.equal(native[3:], pto[3:])
    assert int((native != pto).sum()) == 155


def test_projection_single_factor_directly_explains_static_t6_fix():
    r = read("t6-projection.json.gz")["projection_diagnosis"]
    assert r["native_reproduces_FULL_qli_weights"]
    assert all(r[k]["bit_equal"] for k in ("hc_mixed", "hc_post", "hc_comb", "normed"))
    data = witness("projection-witness.pt.gz")
    qli = witness("qli-witness.pt.gz")["qli_inputs"]["native_FULL"]
    assert torch.equal(data["native_weights"].view(torch.uint8), qli["weights"].view(torch.uint8))
    for name, expected in (("raw", "native_raw"), ("scaled", "native_scaled"), ("weights", "native_weights")):
        assert r["modes"]["0"][name]["different_elements"] == 1
        assert r["modes"]["0"][name]["first_coordinates"] == [[2, 48]]
        assert r["modes"]["1"][name]["bit_equal"]
        assert data["1"][name].numel() == 384
        assert torch.equal(data["1"][name].view(torch.uint8), data[expected].view(torch.uint8))
    assert float(data["0"]["raw"][2, 48]) == 0.578125
    assert float(data["native_raw"][2, 48]) == 0.57421875


def test_actual_kernel_artifacts_obey_descriptor_and_handoff_limits():
    data = archive("generated-static.tar.gz")
    audit = read("kernel-audit-static.json.gz")
    assert audit["receipt_sha256"] == sha(gzip.decompress((EVIDENCE / "compile-static.json.gz").read_bytes()))
    assert all(sha(value) == audit["generated"][name] for name, value in data.items())
    source = ast.parse(data["kernel_config.py"])
    assignments = {n.targets[0].id: n.value for n in source.body if isinstance(n, ast.Assign)}
    kernels = assignments["KERNELS"].elts
    assert len(kernels) == 63
    signatures = []
    for kernel in kernels:
        values = {k.value: v for k, v in zip(kernel.keys, kernel.values)}
        signatures.append(len(values["signature"].elts))
    assert max(signatures) == 32
    orch = data["orchestration/_full_csa_tnd_spmd.cpp"].decode()
    assert "MixedKernels mixed_0 = {0, 1, 1};" in orch
    assert "params_t0.launch_spec.set_block_num(24);" in orch
    assert "params_t0.launch_spec.set_require_sync_start(true);" in orch
    for name in ("kernels/aic/packed_prefix_indexer_cube.cpp", "kernels/aiv/packed_prefix_indexer_vector.cpp"):
        cpp = data[name].decode()
        cube, vector = cpp.split("static __aicore__ void packed_prefix_indexer_vector", 1)
        assert cube.count("SYNCALL<SyncCoreType::Mix>();") == 10
        assert vector.count("SYNCALL<SyncCoreType::Mix>();") == 10
