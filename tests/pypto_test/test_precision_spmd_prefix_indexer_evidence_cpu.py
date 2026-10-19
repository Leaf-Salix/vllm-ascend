# SPDX-License-Identifier: Apache-2.0
"""Bind packed-prefix device results to compiled sources and diagnostic tools."""

import ast
import gzip
import hashlib
import io
import json
import statistics
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd"
EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_prefix_indexer_20261019"


def read_json(name):
    return json.loads(gzip.decompress((EVIDENCE / name).read_bytes()))


def tools():
    with tarfile.open(fileobj=io.BytesIO((EVIDENCE / "tools.tar.gz").read_bytes()), mode="r:gz") as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers()}


def assert_sources(values):
    for name, expected in values.items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected


def test_indexer_prefix_archive_and_compile_identity():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert len(manifest) == 8
    for name, expected in manifest.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    compiled = read_json("compile-packaged.json.gz")
    for gate in (
        "compiled",
        "kernel_mode_abi_pass",
        "control_hc_kernel_mode_abi_pass",
        "control_indexer_kernel_mode_abi_pass",
    ):
        assert compiled[gate] is True
    assert len(compiled["sources"]) == 8
    assert_sources(compiled["sources"])
    archive = tools()
    draft = read_json("compile-packed-draft.json.gz")
    for name, expected in draft["draft_sources"].items():
        assert hashlib.sha256(archive[name]).hexdigest() == expected
    assert_sources(draft["sources"])


@pytest.mark.parametrize(
    ("tokens", "name"),
    [
        (1, "t1-packaged-native-result.json.gz"),
        (21, "t21-draft-native-result.json.gz"),
        (240, "t240-packaged-native-result.json.gz"),
    ],
)
def test_native_prefix_indexer_replay_evidence(tokens, name):
    r = read_json(name)
    assert r["complete"] and r["pass"] and r["tokens"] == tokens
    assert "not vLLM FULL or complete CSA" in r["scope"]
    assert len(r["comparisons"]) == 151
    assert all(c["bit_equal"] and c["different_elements"] == 0 and c["max_abs"] == 0 for c in r["comparisons"].values())
    assert_sources(r["candidate_sources"])
    receipt = read_json("compile-packed-draft.json.gz" if tokens == 21 else "compile-packaged.json.gz")
    assert r["candidate_sources"] == receipt["sources"] and r["control_sources"] == receipt["control_sources"]
    archive = tools()
    helper = "probe_prefix_indexer_packed.py" if tokens == 21 else "probe_prefix_indexer_packaged.py"
    assert hashlib.sha256(archive[helper]).hexdigest() == r["helper_sha256"]
    for phase in ("initial", "A", "B", "A-restored"):
        for stage in ("raw", "scaled", "output"):
            assert r["comparisons"][f"{phase}/control-indexer-vs-native/{stage}"]["bit_equal"]
    for phase in ("A", "B", "A-restored"):
        for repeat in range(3):
            for stage in (
                "mixed",
                "post",
                "comb",
                "normalized",
                "rstd",
                "output",
                "idx_raw",
                "idx_scaled",
                "idx_output",
                "input",
            ):
                assert r["comparisons"][f"{phase}/{repeat}/{stage}"]["bit_equal"]
    assert r["graph_replay_us"]["median"] == statistics.median(r["graph_replay_us"]["samples"])
    if tokens == 21:
        for name, expected in r["draft_sources"].items():
            assert hashlib.sha256(archive[name]).hexdigest() == expected

        def functions(data):
            return [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef)]

        for old, new in [("packed_prefix_indexer_root_draft.py", "hc_prefix_indexer.py")]:
            assert ast.dump(ast.Module(body=functions(archive[old]), type_ignores=[])) == ast.dump(
                ast.Module(body=functions((PACKAGE / new).read_text()), type_ignores=[])
            )
        body = functions(archive["packed_prefix_stages_draft.py"]) + functions(
            archive["indexer_projection_stages_draft.py"]
        )
        assert ast.dump(ast.Module(body=body, type_ignores=[])) == ast.dump(
            ast.Module(body=functions((PACKAGE / "hc_prefix_indexer_stages.py").read_text()), type_ignores=[])
        )


def test_harness_failure_is_not_precision_evidence():
    failure = read_json("t21-harness-failure-result.json.gz")
    assert failure["complete"] is False and failure["pass"] is False
    assert set(failure["comparisons"]) == {"weight-layout-values"}
    archive = tools()
    original = archive["failed-probe-v7/probe_prefix_indexer_packed.py"]
    assert hashlib.sha256(original).hexdigest() == failure["helper_sha256"]
    assert b"impl=layer.self_attn.dsa_attn.impl" in original
    assert b"impl=layer.self_attn.dsa_attn.dsa_attn.impl" in archive["probe_prefix_indexer_packed.py"]
