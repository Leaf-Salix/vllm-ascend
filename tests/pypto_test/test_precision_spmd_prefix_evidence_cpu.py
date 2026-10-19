# SPDX-License-Identifier: Apache-2.0
"""Bind prefix/matched-control device evidence to the actual source identities."""

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
EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_prefix_20261019"


def read_json(name):
    return json.loads(gzip.decompress((EVIDENCE / name).read_bytes()))


def tools():
    with tarfile.open(fileobj=io.BytesIO((EVIDENCE / "tools.tar.gz").read_bytes()), mode="r:gz") as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers()}


def assert_sources(sources):
    for name, expected in sources.items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected


def test_prefix_archives_and_frozen_baseline():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert len(manifest) == 8
    for name, expected in manifest.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    baseline = json.loads(tools()["BASELINE.json"])
    assert len(baseline["baseline_runtime_sources"]) == 43
    for name, expected in baseline["baseline_runtime_sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    compiled = read_json("compile-prefix-packaged.json.gz")
    assert compiled["compiled"] and compiled["kernel_mode_abi_pass"] and compiled["control_hc_kernel_mode_abi_pass"]
    assert len(compiled["sources"]) == 6
    assert_sources(compiled["sources"])


@pytest.mark.parametrize(
    ("tokens", "filename"),
    [
        (1, "t1-packaged-native-result.json.gz"),
        (21, "t21-draft-native-result.json.gz"),
        (240, "t240-packaged-native-result.json.gz"),
    ],
)
def test_prefix_native_and_graph_evidence(tokens, filename):
    result = read_json(filename)
    assert result["complete"] and result["pass"] and result["tokens"] == tokens
    assert "not vLLM FULL or complete CSA" in result["scope"]
    assert len(result["comparisons"]) == 103
    assert all(
        c["bit_equal"] and c["different_elements"] == 0 and c["max_abs"] == 0 for c in result["comparisons"].values()
    )
    assert_sources(result["candidate_sources"])
    for phase in ("initial", "A", "B", "A-restored"):
        for stage in ("mixed", "post", "comb"):
            assert result["comparisons"][phase + "/control-hc-vs-native/" + stage]["bit_equal"]
    for phase in ("A", "B", "A-restored"):
        for repeat in range(3):
            for stage in ("mixed", "post", "comb", "normalized", "rstd", "output", "input"):
                assert result["comparisons"][f"{phase}/{repeat}/{stage}"]["bit_equal"]
    assert result["graph_replay_us"]["median"] == statistics.median(result["graph_replay_us"]["samples"])
    archive = tools()
    assert (
        hashlib.sha256(archive["probe_prefix.py" if tokens == 21 else "probe_prefix_packaged.py"]).hexdigest()
        == result["helper_sha256"]
    )
    if tokens == 21:
        for name, expected in result["draft_sources"].items():
            assert hashlib.sha256(archive[name]).hexdigest() == expected
        for old, new in [("prefix_root_draft.py", "hc_prefix.py"), ("prefix_stages_draft.py", "hc_prefix_stages.py")]:
            before = [n for n in ast.parse(archive[old]).body if isinstance(n, ast.FunctionDef)]
            after = [n for n in ast.parse((PACKAGE / new).read_text()).body if isinstance(n, ast.FunctionDef)]
            assert ast.dump(ast.Module(body=before, type_ignores=[])) == ast.dump(
                ast.Module(body=after, type_ignores=[])
            )


def test_matched_single_entry_performance_and_precision():
    result = read_json("t240-matched-control-result.json.gz")
    assert result["complete"] and result["pass"] and result["tokens"] == 240
    assert len(result["comparisons"]) == 176 and all(c["bit_equal"] for c in result["comparisons"].values())
    assert_sources(result["candidate_sources"])
    archive = tools()
    assert hashlib.sha256(archive["probe_slice_matched_perf.py"]).hexdigest() == result["helper_sha256"]
    compiled = read_json("compile-matched-control.json.gz")
    assert compiled["compiled"] and compiled["kernel_mode_abi_pass"]
    assert compiled == result["matched_control_receipt"]
    assert hashlib.sha256(archive["norm_wkv_orchestrated_control.py"]).hexdigest() == compiled["source_sha256"]
    assert result["control_sources"] == compiled["control_sources"]
    names = ["native", "two_entry_control", "single_entry_control", "spmd"]
    assert result["timing_order"] == [names[i:] + names[:i] for i in range(4)] * 2
    for arm in names:
        samples = result["graph_replay_us"][arm]["samples"]
        assert len(samples) == 8 and all(v > 0 for v in samples)
        assert result["graph_replay_us"][arm]["median"] == statistics.median(samples)
        for phase in ("A", "B", "A-restored"):
            for stage in ("norm", "wkv", "input"):
                assert result["comparisons"][f"graphs/{phase}/{arm}/{stage}"]["bit_equal"]
