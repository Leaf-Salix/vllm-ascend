# SPDX-License-Identifier: Apache-2.0
"""Current-source/NPU-evidence binding and CPU replay of whole-model token checks.

This verifies recorded acceptance evidence; it does not execute PTO numerics.
"""

import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/current_full_precision"
_PACKAGE_PARENT = "vllm_ascend/ops/pypto"
_PACKAGES = ("deepseek_v4_flash_csa_tnd_precision", "deepseek_v4_flash_csa_tnd_full_precision")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _artifact(record):
    path = (_EVIDENCE / record["path"]).resolve()
    assert path.is_relative_to(_EVIDENCE.resolve()), "Evidence path escapes its directory"
    data = path.read_bytes()
    assert _sha(data) == record["sha256"], path
    return data


def _manifest():
    data = json.loads((_EVIDENCE / "manifest.json").read_text())
    assert data["schema_version"] == 2 and data["status"] == "accepted"
    return data


def test_current_runtime_sources_match_actual_npu_snapshot():
    m = _manifest()
    required = {
        str(path.relative_to(_ROOT))
        for package in _PACKAGES
        for path in (_ROOT / _PACKAGE_PARENT / package).rglob("*.py")
    }
    required.update(f"{_PACKAGE_PARENT}/{name}" for name in ("variant.py",))
    required.add("vllm_ascend/ops/dsv4_csa.py")
    assert required == set(m["sources"]), "Current complete runtime source set differs"
    for name, digest in m["sources"].items():
        assert _sha((_ROOT / name).read_bytes()) == digest, name
    assert len(m["whole_model"]) == 2
    assert {run["case"] for run in m["whole_model"]} == {"8k", "128k"}
    for run in m["whole_model"]:
        archive = _EVIDENCE / run["archive"]["path"]
        _artifact(run["archive"])
        with tarfile.open(archive) as bundle:
            source_raw = bundle.extractfile(run["source_manifest_member"]).read()
            config = json.load(bundle.extractfile("results/configuration.json"))
            helper_raw = bundle.extractfile("helpers/helper-manifest.json").read()
            assert _sha(helper_raw) == config["helper_manifest_sha256"]
            for name, digest in json.loads(helper_raw)["files"].items():
                assert _sha(bundle.extractfile("helpers/" + name).read()) == digest
        assert _sha(source_raw) == config["source_manifest_sha256"] == m["source_manifest_sha256"]
        tested = json.loads(source_raw)["files"]
        assert all(tested.get(name) == digest for name, digest in m["sources"].items())
        assert config["single_pto_entry"] and config["handwritten_cpp"] is False
        assert config["variant"] == "tnd_full_precision"


@pytest.mark.parametrize("case", ("8k", "128k"))
def test_full_model_saved_tokens_events_and_replacement_qualification(case):
    m = _manifest()
    run = next(run for run in m["whole_model"] if run["case"] == case)
    _artifact(run["archive"])
    expected_batch, bucket = (40, 240) if case == "8k" else (24, 144)
    with tarfile.open(_EVIDENCE / run["archive"]["path"]) as bundle:
        read = lambda name: json.load(bundle.extractfile(name))
        cfg = read("results/configuration.json")
        assert cfg["case"] == case and cfg["backend"] == "pto"
        assert cfg["runtime_manifest_sha256"] == m["runtime_manifest_sha256"]
        assert cfg["batch"] == expected_batch and cfg["decode_tokens"] == 192
        assert cfg["capture_sizes"] == [6, bucket] and cfg["warmup_tokens"] == 192
        assert cfg["repeats"] >= 1
        layers = {f"model.layers.{i}.self_attn.attn" for i in range(2, 43, 2)}
        for rank in range(16):
            actual = read(f"results/rank{rank}.real_perf.json")
            native = read(f"gold/rank{rank}.mixed.json")
            native_tokens = {r["key"]: r["output_token_ids"] for r in native["requests"]}
            native_events = {
                r["key"]: r["events"]
                for r in map(json.loads, bundle.extractfile(f"gold/request_stats_rank{rank}.jsonl"))
            }
            assert actual["rank"] == rank and actual["batch"] == expected_batch
            assert actual["backend"] == "pto" and actual["variant"] == "tnd_full_precision"
            assert len(actual["implementations"]) == len(actual["worker_runtime_config"]) == 1
            for worker in actual["implementations"]:
                csa = [layer for layer in worker if layer["kind"] == "csa"]
                hca = [layer for layer in worker if layer["kind"] == "hca"]
                assert len(csa) == 21 and len(hca) == 20
                assert all(layer["implementation"] == "pto" for layer in csa)
                assert all(layer["implementation"] == "native" for layer in hca)
            assert actual["worker_runtime_config"]
            for worker in actual["worker_runtime_config"]:
                assert worker["deterministic_level"] == 1 and worker["torch_deterministic"]
                assert worker["hccl_deterministic"] == "true" and not worker["dynamic_eplb"]
                assert worker["cann_event_work_mode"] == 1
            assert len(actual["rounds"]) == cfg["repeats"]
            for phase in [actual["warmup"], *actual["rounds"]]:
                requests = phase["requests"]
                assert len(requests) == expected_batch
                assert len({r["key"] for r in requests}) == expected_batch
                assert {r["key"] for r in requests} == set(native_tokens)
                for request in requests:
                    assert request["output_token_ids"] == native_tokens[request["key"]]
                    assert len(request["output_token_ids"]) == 192
                    assert request["events"] == native_events[request["key"]]
                    assert request["preemptions"] == 0
            for row in actual["rounds"]:
                assert len(row["timing"]) == 1
                q = row["timing"][0]["identity"]["static_full_qualification"]
                assert q["backend"] == "pto" and q["ready"]
                assert tuple(q["restored_modules"]) == (
                    "hc_pre",
                    "input_norm",
                    "wkv",
                    "indexer_weights",
                    "o_proj",
                    "hc_post",
                )
                assert q["metadata_sha256"] == cfg["static_metadata_sha256"]
                assert q["actual_reselections"] == 1 and q["validated_reuses"] > 0
                assert q["root_namespace"] == "dsv4_csa_tnd_full_precision::attention"
                assert q["root_abi_tensors"] == 62 and set(q["single_root_counts"]) == layers
                for counts in q["single_root_counts"].values():
                    for size in (6, bucket):
                        assert counts[str(size)]["warmup"] > 0 and counts[str(size)]["capture"] > 0


def test_formal_mixed_csa_raw_precision_is_source_bound():
    """Recheck actual formal FULL-graph records; this does not execute NPU math."""
    m = _manifest()
    item = m["csa_mixed"]
    _artifact(item["archive"])
    with tarfile.open(_EVIDENCE / item["archive"]["path"]) as bundle:
        raw = json.load(bundle.extractfile(item["result_member"]))
        source_raw = bundle.extractfile(item["source_manifest_member"]).read()
        helpers_raw = bundle.extractfile(item["helper_manifest_member"]).read()
        helpers = json.loads(helpers_raw)
        assert _sha(helpers_raw) == raw["manifest_sha256"]
        assert helpers == raw["helper_sha"]
        for name, digest in helpers.items():
            assert _sha(bundle.extractfile(item["helper_prefix"] + name).read()) == digest
        assert _sha(source_raw) == raw["source_manifest_sha256"] == m["source_manifest_sha256"]
        snapshot = json.loads(source_raw)["files"]
        assert raw["tested_sources"] == m["sources"]
        assert all(snapshot.get(name) == digest for name, digest in m["sources"].items())
        for name, digest in m["sources"].items():
            assert _sha(bundle.extractfile(item["source_prefix"] + name).read()) == digest
    assert raw["pass"] is True and raw["complete"] is True
    assert raw["lengths"] == [1, 2, 3, 4, 5, 6]
    assert raw["true_tokens"] == raw["physical_graph_tokens"] == 21
    assert raw["per_request_padding"] == 0 and raw["whole_model_sitecustomize"] is False
    assert raw["static"]["actual_reselect_count"] == 1
    assert raw["static"].get("validated_reuses", 0) > 0
    assert raw["static"]["metadata_sha256"] == "863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602"
    assert raw["hc_weight_qualification"]["version"] == 1
    for name in ("raw_sha256", "rounded_sha256"):
        assert len(raw["hc_weight_qualification"][name]) == 64
    expected = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0", "indexer.0", "indexer.1", "indexer_state.0"}
    for phase in ("comparisons", "B_comparisons"):
        values = raw[phase]["candidate"]
        assert set(values) == expected
        for result in values.values():
            assert result["bit_equal"] is True and result["different_elements"] == 0
            assert result["max_abs"] == result["relative_l2"] == 0
    for arm in ("native", "candidate"):
        graph = raw["graphs"][arm]
        assert graph["actual_acl_graph"] and graph["fullgraph"]
        assert graph["compiled_raw_equal"] and graph["topk_poison_replay_restored"] and graph["live_input_ABA"]
        assert graph["true_tokens"] == graph["graph_tokens"] == 21
        assert all(g["status"] == "PASS" for g in graph["guards"].values())
    assert raw["graphs"]["candidate"]["pto_dispatch_calls"] > 0
    schema = raw["graphs"]["candidate"]["root_schema"]
    namespace = item["root_namespace"]
    assert namespace.startswith("formal_csa_") and namespace.endswith("::attention")
    assert schema.startswith(namespace + "(")
    assert schema.split(" -> ")[0].count("Tensor") == 62
    assert raw["entry"]["candidate"]["eligible"] and raw["entry"]["candidate"]["route"] == "PTO"
    assert raw["timing_executed"] is True
    assert all(len(values) == 8 for values in raw["timings"]["samples"].values())


@pytest.mark.parametrize("case", ("8k", "128k"))
def test_saved_CPU_workload_geometry_and_maxrank_timing(case):
    """CPU buffers are observations, not GPU input hooks or new hardware timing."""
    m = _manifest()
    run = next(run for run in m["whole_model"] if run["case"] == case)
    _artifact(run["archive"])
    candidate, native = [], []
    with tarfile.open(_EVIDENCE / run["archive"]["path"]) as bundle:
        cfg = json.load(bundle.extractfile("results/configuration.json"))
        native_cfg = json.load(bundle.extractfile("native_perf/configuration.json"))
        for field in (
            "bank_plan_sha256",
            "runtime_manifest_sha256",
            "bank",
            "blocks",
            "capture_sizes",
            "batch",
            "decode_tokens",
            "warmup_tokens",
            "static_metadata_sha256",
        ):
            assert cfg[field] == native_cfg[field], field
        for rank in range(16):
            actual = json.load(bundle.extractfile(f"results/rank{rank}.real_perf.json"))
            control = json.load(bundle.extractfile(f"native_perf/rank{rank}.real_perf.json"))
            gold = json.load(bundle.extractfile(f"gold/rank{rank}.mixed.json"))
            gold_ids = {r["key"]: r["output_token_ids"] for r in gold["requests"]}
            event_gold = {
                r["key"]: r["events"]
                for r in map(json.loads, bundle.extractfile(f"gold/request_stats_rank{rank}.jsonl"))
            }
            for round_record in actual["rounds"]:
                timing = round_record["timing"][0]
                before = round_record["identity_before"][0]["cache_stats"]
                after = timing["identity"]["cache_stats"]
                for field in (
                    "generation_builds",
                    "binary_builds",
                    "bypasses",
                    "invalid_entries",
                    "storage_errors",
                    "forced_rebuilds",
                ):
                    assert before[field] == after[field]
                for field in ("bypasses", "invalid_entries", "storage_errors", "forced_rebuilds"):
                    assert before[field] == after[field] == 0
                left_rows = timing["metadata"]
                assert len(left_rows) == 20
                for native_round in control["rounds"]:
                    assert {r["key"]: r["output_token_ids"] for r in native_round["requests"]} == gold_ids
                    assert {r["key"]: r["events"] for r in native_round["requests"]} == event_gold
                    right_rows = native_round["timing"][0]["metadata"]
                    assert len(right_rows) == 20
                    for left, right in zip(left_rows, right_rows, strict=True):
                        omitted = (
                            {"call"} if case == "8k" or native_round["iteration"] == 1 else {"call", "input_ids_cpu"}
                        )
                        assert {k: v for k, v in left.items() if k not in omitted} == {
                            k: v for k, v in right.items() if k not in omitted
                        }
                        if case == "128k":
                            assert control["rounds"][0]["iteration"] == 1
                            assert len(left["input_ids_cpu"]) == len(right["input_ids_cpu"]) == left["actual_tokens"]
                            # Later Native windows have known CPU zero/-1 sentinel
                            # differences inside this observer slice. Never label
                            # these as proof of GPU padding or GPU input equality.
                            for a, b in zip(left["input_ids_cpu"], right["input_ids_cpu"], strict=True):
                                assert a == b or {a, b} == {0, -1}
            candidate.append(actual)
            native.append(control)

    def maxrank_ms(records, kind):
        values = []
        for iteration in range(len(records[0]["rounds"])):
            samples = [r["rounds"][iteration]["timing"][0][kind]["samples_us"] for r in records]
            count = 20 if kind == "forward" else 19
            assert all(len(row) == count and all(math.isfinite(v) and v > 0 for v in row) for row in samples)
            values.extend(max(row[step] for row in samples) for step in range(count))
        return statistics.mean(values) / 1000

    for label, records in (("candidate", candidate), ("historical_native", native)):
        for kind in ("forward", "cycle"):
            assert run["recomputed"][label + "_" + kind + "_ms"] == pytest.approx(maxrank_ms(records, kind), rel=1e-12)
