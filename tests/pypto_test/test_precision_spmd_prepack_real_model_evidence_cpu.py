# SPDX-License-Identifier: Apache-2.0
"""Recompute whole-model token, workload and timing claims from archived raw data."""

import hashlib
import json
import statistics
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_prepack_real_model_20261020"
SUMMARY = json.loads((EVIDENCE / "summary.json").read_text())


@pytest.fixture(params=["8k", "128k"])
def raw(request):
    case = request.param
    with tarfile.open(EVIDENCE / f"{case}-raw.tar.gz") as archive:
        data = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    return case, data


def read(data, name):
    return json.loads(data[name])


def test_archive_integrity_and_exact_runtime_sources(raw):
    case, data = raw
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
    audit = read(data, "audit.json")
    assert audit["pass"] and audit["source_files_rehashed"]
    assert hashlib.sha256(data["source-manifest.json"]).hexdigest() == SUMMARY[case]["source_manifest_sha256"]
    assert len(read(data, "source-manifest.json")["files"]) == 3385
    assert len(audit["tested_sources"]) == 63
    for name, digest in audit["tested_sources"].items():
        assert read(data, "source-manifest.json")["files"][name] == digest
        assert hashlib.sha256(data[f"runtime-sources/{name}"]).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_all_ranks_all_windows_match_saved_native_tokens_and_events(raw):
    case, data = raw
    tokens = events = 0
    for rank in range(16):
        actual = read(data, f"candidate/rank{rank}.real_perf.json")
        gold = read(data, f"native-gold/rank{rank}.mixed.json")
        expected = {entry["key"]: entry for entry in gold["requests"]}
        event_gold = {
            row["key"]: row["events"]
            for row in map(json.loads, data[f"native-gold/request_stats_rank{rank}.jsonl"].splitlines())
        }
        assert actual["rank"] == rank and len(actual["rounds"]) == 3
        for window in [actual["warmup"], *actual["rounds"]]:
            assert len(window["requests"]) == len(expected)
            assert {entry["key"] for entry in window["requests"]} == set(expected)
            for entry in window["requests"]:
                reference = expected[entry["key"]]
                assert entry["output_token_ids"] == reference["output_token_ids"]
                assert entry["events"] == event_gold[entry["key"]]
                assert entry["preemptions"] == 0
                assert len(entry["output_token_ids"]) == 192
                tokens += len(entry["output_token_ids"])
                events += len(entry["events"])
    assert (tokens, events) == ((491520, 105152) if case == "8k" else (294912, 61568))
    assert tokens == SUMMARY[case]["tokens_equal_native"]
    assert events == SUMMARY[case]["accept_events_equal_native"]


def test_measured_workload_strictly_matches_same_native_phase(raw):
    case, data = raw
    count = 0
    for rank in range(16):
        actual = read(data, f"candidate/rank{rank}.real_perf.json")
        native = read(data, f"native-historical/rank{rank}.real_perf.json")
        assert [row["iteration"] for row in actual["rounds"]] == [1, 2, 3]
        assert [row["iteration"] for row in native["rounds"]] == ([1, 2, 3] if case == "8k" else [1, 2])
        for index, window in enumerate(actual["rounds"]):
            references = native["rounds"] if case == "8k" else [native["rounds"][min(index, 1)]]
            for reference in references:
                rows = window["timing"][0]["metadata"]
                gold = reference["timing"][0]["metadata"]
                assert len(rows) == len(gold) == 20
                for left, right in zip(rows, gold):
                    assert {k: v for k, v in left.items() if k != "call"} == {
                        k: v for k, v in right.items() if k != "call"
                    }
                    assert left["graph_mode"] == "FULL"
                    assert left["requests"] == (40 if case == "8k" else 21)
                    assert left["actual_tokens"] == (240 if case == "8k" else 126)
                    assert left["padded_tokens"] == (240 if case == "8k" else 144)
                    count += 1
    assert count == (2880 if case == "8k" else 960)


def test_timing_recomputed_without_outlier_removal(raw):
    case, data = raw
    for arm in ("candidate", "native-historical", "frozen-precision", "spmd-parent"):
        ranks = [read(data, f"{arm}/rank{rank}.real_perf.json") for rank in range(16)]
        for kind, width in (("forward", 20), ("cycle", 19)):
            samples = []
            means = []
            for index in range(len(ranks[0]["rounds"])):
                rows = [rank["rounds"][index]["timing"][0][kind]["samples_us"] for rank in ranks]
                assert all(len(row) == width for row in rows)
                window = [max(row[step] for row in rows) / 1000 for step in range(width)]
                samples.extend(window)
                means.append(statistics.mean(window))
            expected = SUMMARY[case]["metrics"][arm][kind]
            assert samples == expected["samples_ms"]
            assert means == expected["round_means_ms"]
            assert statistics.mean(samples) == expected["mean_ms"]
            assert statistics.pstdev(samples) == expected["pstdev_ms"]
            assert max(samples) == expected["max_ms"]


def test_formal_preparer_and_full_probe_bind_same_sources(raw):
    _, data = raw
    receipt = read(data, "formal-prepack-compile.json")
    layer = read(data, "single-layer-full-lifecycle.json")
    audit = read(data, "audit.json")
    assert layer["complete"] and layer["pass"]
    assert layer["candidate_sources"] == receipt["sources"]
    assert layer["compile_receipt_sha256"] == hashlib.sha256(data["formal-prepack-compile.json"]).hexdigest()
    assert len(receipt["sources"]) == 6
    assert receipt["formal_prepare_program_compiled"] and receipt["formal_prepare_kernel_mode_abi_pass"]
    assert receipt["formal_operators_registered"]
    for name, digest in receipt["sources"].items():
        relative = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd_prepack/" + name
        assert audit["tested_sources"][relative] == digest
