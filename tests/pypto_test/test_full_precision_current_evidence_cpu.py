# SPDX-License-Identifier: Apache-2.0
"""Current-source/NPU-evidence binding and CPU replay of whole-model token checks.

This verifies recorded acceptance evidence; it does not execute PTO numerics.
"""

import ast
import csv
import hashlib
import io
import json
import math
import runpy
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
    phase = data.get("reference_phase_policy")
    if phase is not None:
        assert phase["candidate_to_native_iteration"] == {"1": 1, "2": 2, "3": 2}
        assert phase["excluded_metadata_fields"] == ["call"]
        assert phase["input_ids_cpu_normalization"] is False
        _artifact(phase["archive"])
        with tarfile.open(_EVIDENCE / phase["archive"]["path"]) as bundle:
            read = lambda name: bundle.extractfile(name).read()
            policy = json.loads(read("phase-policy.json"))
            for field in ("candidate_to_native_iteration", "excluded_metadata_fields", "input_ids_cpu_normalization"):
                assert policy[field] == phase[field]
            assert _sha(read("original-independent128k.py")) == policy["original_strict_auditor_sha256"]
            assert _sha(read("original-strict-failure.log")) == policy["original_strict_failure_sha256"]
            assert _sha(read("independent-review.json")) == policy["independent_review_sha256"]
            audit = json.loads(read("phase-audit.json"))
            assert audit["pass"] and audit["source_manifest_sha256"] == data["source_manifest_sha256"]
            assert audit["auditor_sha256"] == _sha(read("independent128k_phase_matched.py"))
            assert audit["reference_phase_policy_sha256"] == _sha(read("phase-policy.json"))
            assert audit["reference_phase_policy"] == policy
            fixtures = json.loads(read("phase-fixture-results.json"))
            assert fixtures["pass"] and fixtures["normalization_used"] is False
            assert len(fixtures["checks"]) == 7
            assert {item["case"]: item["expected_success"] for item in fixtures["checks"]} == {
                "unmodified_copy_of_actual_three_rounds": True,
                "first_phase_input_0_to_negative1": False,
                "repeated_phase2_input_negative1_to0": False,
                "repeated_phase3_input_negative1_to0": False,
                "non_input_geometry_changed": False,
                "missing_final_round": False,
                "final_round_output_token_changed": False,
            }
            assert all((item["actual_exit"] == 0) == item["expected_success"] for item in fixtures["checks"])
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
        assert cfg["repeats"] == m.get("expected_repeats", 1)
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
            assert [row["iteration"] for row in actual["rounds"]] == list(range(1, cfg["repeats"] + 1))
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
                reference_iteration = 1
                if case == "128k" and "reference_phase_policy" in m:
                    reference_iteration = 1 if round_record["iteration"] == 1 else 2
                    assert [row["iteration"] for row in control["rounds"]] == [1, 2]
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
                            {"call"}
                            if case == "8k" or native_round["iteration"] == reference_iteration
                            else {"call", "input_ids_cpu"}
                        )
                        assert {k: v for k, v in left.items() if k not in omitted} == {
                            k: v for k, v in right.items() if k not in omitted
                        }
                        if case == "128k":
                            assert control["rounds"][0]["iteration"] == 1
                            assert len(left["input_ids_cpu"]) == len(right["input_ids_cpu"]) == left["actual_tokens"]
                            # Unmatched measured phases have known CPU zero/-1 sentinel
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


def test_recorded_qkpv_single_factor_precision_and_paired_timings():
    """Replay recorded provenance/timing arithmetic; does not execute PTO math."""
    m = _manifest()
    item = m["qkpv_swa"]
    _artifact(item["archive"])
    with tarfile.open(_EVIDENCE / item["archive"]["path"]) as bundle:
        raw = lambda name: bundle.extractfile(name).read()
        records = json.loads(raw("ARCHIVE_MANIFEST.json"))["files"]
        for name, record in records.items():
            assert _sha(raw(name)) == record["sha256"]
        candidate_bytes = raw("candidate-swa-cube/candidate_sparse.py")
        candidate_sha = "8f3ddd066cbf4f7111b554baaa0f3720f940c5db96bb32235cecdfad201ea20c"
        assert _sha(candidate_bytes) == candidate_sha
        candidate = candidate_bytes.decode()
        candidate = candidate.replace(
            "from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.config import (",
            "from .config import (",
        ).replace(
            "from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.layout import ",
            "from .layout import ",
        )
        # Preserved SWA128 checkpoint; the live-staging test binds current code.
        formal = raw("formatted_candidate_sparse.py")
        assert _sha(formal) == "248d8b11bb0a045f7863c0b264d962845dcabafa51cbf1e5fafee413fa8dc6d5"
        assert ast.dump(ast.parse(candidate)) == ast.dump(ast.parse(formal))
        fields = ("mi", "li", "oi", "heads")
        labels = {"Q_B_actual_input", "baseline.Native_heads", "candidate.Native_heads"}
        labels.update(f"device_readonly.{i}" for i in range(12))
        labels.update(
            f"{arm}.{phase}.{field}"
            for arm in ("baseline", "candidate")
            for phase in ("raw_replay", "A_restored", "final_gold")
            for field in fields
        )
        labels.update(
            f"{phase}.{field}"
            for phase in (
                "A",
                "B_Q_changed",
                "Q_zero",
                "A_restored",
                "timing_round4",
                "timing_round9",
                "timing_round14",
                "timing_round19",
            )
            for field in fields
        )
        boundaries = {
            "x_out",
            "idx_topk",
            "swa.0",
            "compressed.0",
            "state.0",
            "indexer.0",
            "indexer.1",
            "indexer_state.0",
        }
        for batch, capture_path in (
            (4, "capture-b4-results/data/result.json"),
            (40, "capture-b40-results/result.json"),
        ):
            capture = json.loads(raw(capture_path))
            assert capture["pass"] and capture["complete"]
            assert capture["true_tokens"] == batch * 6
            assert set(capture["capture_matches_Native_FULL"]) == boundaries
            assert all(row["bit_equal"] for row in capture["capture_matches_Native_FULL"].values())
            for repetition in (1, 2):
                prefix = "sparse-graph" + ("-b40" if batch == 40 else "")
                result = json.loads(raw(f"{prefix}-results-r{repetition}/data/result.json"))
                assert result["pass"] and result["complete"]
                helper_dir = "sparse-graph-ab" + ("-b40" if batch == 40 else "")
                assert _sha(raw(helper_dir + "/manifest.json")) == result["helper_manifest_sha"]
                for provider in ("baseline_root", "candidate_root", "candidate_sparse"):
                    assert (
                        _sha(raw(helper_dir + "/" + provider + ".py")) == result["source_modules"][provider]["sha256"]
                    )
                expected_stats = (
                    "eb43040864b3b1209d1533f8ce0f2de68b6cee6be322905f45fba6b44e82feb2"
                    if batch == 4
                    else "85b7fc7845dfe2b56e088db861c1277a26d0da16b233db25c73670c5a065bf1f"
                )
                assert result["stats_sha"] == expected_stats
                assert result["source_modules"]["candidate_sparse"]["sha256"] == candidate_sha
                baseline_source = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.decode_sparse_attn_csa"
                assert (
                    result["source_modules"][baseline_source]["sha256"]
                    == "f5558603e25711de2a0ae9422dc62de977c36159c668f13d4451b176b9ca17c6"
                )
                assert {row["label"] for row in result["checks"]} == labels
                assert not result["PMU"] and not result["DFX"] and not result["depgen"]
                assert result["bank_sha"] == capture["captured_bank"]["sha256"]
                assert len(result["checks"]) == 71 and all(row["bit_equal"] for row in result["checks"])
                assert result["timing_rounds"] == 20 and result["replays_per_round"] == 100
                baseline = result["timing_us"]["baseline"]
                optimized = result["timing_us"]["candidate"]
                assert len(baseline["samples"]) == len(optimized["samples"]) == 20
                for arm in (baseline, optimized):
                    assert all(math.isfinite(x) and x > 0 for x in arm["samples"])
                    assert arm["mean"] == pytest.approx(statistics.mean(arm["samples"]), rel=1e-12)
                gains = [100 * (a - b) / a for a, b in zip(baseline["samples"], optimized["samples"], strict=True)]
                assert all(value > 0 for value in gains)
                assert result["paired_speedup_percent"] == pytest.approx(gains, rel=1e-12)


def test_swa_reproduction_profile_is_bound_to_actual_npu_package():
    """The public replay command must select the recorded source and helpers."""
    current = _manifest()
    m = json.loads(_artifact(current["qkpv_swa"]["baseline_manifest"]))
    tool = _ROOT / "tests/pypto_test/prepare_tnd_precision_performance_repro.py"
    profile = runpy.run_path(str(tool))["PROFILES"]["swa128"]
    assert profile["source_sha"] == m["source_manifest_sha256"]
    for run in m["whole_model"]:
        _artifact(run["archive"])
        with tarfile.open(_EVIDENCE / run["archive"]["path"]) as bundle:
            cfg = json.load(bundle.extractfile("results/configuration.json"))
        assert profile["helper_sha"] == cfg["helper_manifest_sha256"]
        assert profile["old_job"] == str(Path(cfg["snapshot"]).parents[1])
        assert profile["job_expression"] == profile["old_job"]


def test_recorded_live_staging_precision_counters_and_paired_timings():
    """Replay raw evidence, preserving all gates; this does not execute PTO math.

    This long replay deliberately binds the frozen source, program counters and
    graph evidence together so successful fragments cannot be mixed across runs.
    """
    m = _manifest()
    item = m["qkpv_live_staging"]
    _artifact(item["archive"])
    sparse = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision/decode_sparse_attn_csa.py"
    history = json.loads(_artifact(m["qkpv_swa"]["baseline_manifest"]))
    assert history["sources"][sparse] == "248d8b11bb0a045f7863c0b264d962845dcabafa51cbf1e5fafee413fa8dc6d5"
    assert {k: v for k, v in m["sources"].items() if k != sparse} == {
        k: v for k, v in history["sources"].items() if k != sparse
    }
    with tarfile.open(_EVIDENCE / m["qkpv_swa"]["archive"]["path"]) as old:
        old_result = json.load(old.extractfile("sparse-graph-results-r1/data/result.json"))
        expected_graph_labels = {row["label"] for row in old_result["checks"]}
    assert len(expected_graph_labels) == 71
    with tarfile.open(_EVIDENCE / item["archive"]["path"]) as bundle:
        raw = lambda name: bundle.extractfile(name).read()
        records = json.loads(raw("ARCHIVE_MANIFEST.json"))["files"]
        assert len(records) == 105
        for name, record in records.items():
            data = raw(name)
            assert _sha(data) == record["sha256"] and len(data) == record["bytes"]
            data.decode("utf8")
        assert _sha(raw("baseline_sparse.py")) == history["sources"][sparse]
        formal = raw("formatted_live_sparse_v2.py")
        assert _sha(formal) == m["sources"][sparse] == _sha((_ROOT / sparse).read_bytes())
        private = raw("helpers/live_staging_sparse.py")
        assert _sha(private) == "36a3b7ebb2ab42ed4dd8c081f66b2ea60a07d40c55ba021da8552f4de5f2ab12"
        normalized = (
            private.decode()
            .replace(
                "from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.config import (",
                "from .config import (",
            )
            .replace(
                "from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.layout import ", "from .layout import "
            )
        )
        assert ast.dump(ast.parse(normalized)) == ast.dump(ast.parse(formal))
        manifest = json.loads(raw("helpers/manifest.json"))
        for name, digest in manifest.items():
            assert _sha(raw("helpers/" + name)) == digest
        gates = json.loads(raw("graph/micro-gates.json"))
        assert gates["pass"] and len(gates["graph_runs"]) == 4
        assert sorted(x["batch"] for x in gates["graph_runs"]) == [4, 4, 40, 40]
        assert len({x["label"] for x in gates["graph_runs"]}) == 4
        assert len({x["result_sha256"] for x in gates["graph_runs"]}) == 4
        for gate in gates["graph_runs"]:
            result_bytes = raw("graph/results/" + gate["label"] + "/data/result.json")
            assert _sha(result_bytes) == gate["result_sha256"]
            r = json.loads(result_bytes)
            assert r["pass"] and r["complete"] and r["task_device"] == "9"
            assert not r["DFX"] and not r["PMU"] and not r["depgen"]
            assert len(r["checks"]) == 71 and {row["label"] for row in r["checks"]} == expected_graph_labels
            assert all(row["bit_equal"] for row in r["checks"])
            assert _sha(raw("helpers/manifest.json")) == r["helper_manifest_sha"]
            for module in ("baseline_root", "candidate_root", "live_staging_sparse", "baseline_sparse"):
                assert _sha(raw("helpers/" + module + ".py")) == r["source_modules"][module]["sha256"]
            assert (
                r["source_modules"]["vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.decode_sparse_attn_csa"][
                    "sha256"
                ]
                == history["sources"][sparse]
            )
            assert r["timing_rounds"] == 20 and r["replays_per_round"] == 100
            a, b = r["timing_us"]["baseline"], r["timing_us"]["candidate"]
            assert len(a["samples"]) == len(b["samples"]) == 20
            for arm in (a, b):
                assert all(math.isfinite(x) and x > 0 for x in arm["samples"])
                assert arm["mean"] == pytest.approx(statistics.mean(arm["samples"]), rel=1e-12)
            differences = [x - y for x, y in zip(a["samples"], b["samples"], strict=True)]
            lower = statistics.mean(differences) - 2.093 * statistics.stdev(differences) / math.sqrt(20)
            assert lower > 0 and sum(x > 0 for x in differences) >= 15
            assert gate["paired_ci_lower_us"] == pytest.approx(lower, rel=1e-12)
            assert gate["positive_pairs"] == sum(x > 0 for x in differences)
            expected_stats = (
                "eb43040864b3b1209d1533f8ce0f2de68b6cee6be322905f45fba6b44e82feb2"
                if gate["batch"] == 4
                else "85b7fc7845dfe2b56e088db861c1277a26d0da16b233db25c73670c5a065bf1f"
            )
            assert r["stats_sha"] == expected_stats
            expected_bank = (
                "66158da293867dc48626083df47404b919ee859fb9719c8d55c0f88f4809668e"
                if gate["batch"] == 4
                else "ee36fbef78c0ad255408fe7796e212eabb1b45ccc29070214ade2110c1b27b8a"
            )
            assert r["bank_sha"] == expected_bank
        program_labels = set()
        for iteration in range(3):
            program_labels.add(f"run{iteration}.Native.pre_inverse_heads")
            program_labels.update(f"run{iteration}.baseline_PTO.{field}" for field in ("mi", "li", "oi", "heads"))
            program_labels.update(f"run{iteration}.readonly.{index}" for index in range(12))
            if iteration:
                program_labels.update(f"run{iteration}.repeat.{field}" for field in ("mi", "li", "oi", "heads"))
        for prefix, variant, physical_card in (
            ("pmu/results/pmu-baseline", "baseline", "9"),
            ("pmu/results/pmu-read-only", "read_only", "9"),
            ("pmu/results/pmu-live-staging", "live_staging", "8"),
            ("pmu-card9", "live_staging", "9"),
        ):
            r = json.loads(raw(prefix + "/data/result.json"))
            assert r["pass"] and r["complete"] and len(r["checks"]) == len(program_labels) == 59
            assert {row["label"] for row in r["checks"]} == program_labels and all(
                row["bit_equal"] for row in r["checks"]
            )
            assert r["bank_sha"] == "66158da293867dc48626083df47404b919ee859fb9719c8d55c0f88f4809668e"
            assert r["helper_manifest_sha256"] == _sha(raw("helpers/manifest.json"))
            assert all(row["different_bytes"] == 0 for row in r["checks"])
            assert [x["pmu"] for x in r["runs"]] == [0, 0, 2] and r["variant"] == variant
            receipt = json.loads(raw("generated/" + variant + "-compiled.json"))
            assert r["artifact_manifest"] == receipt and r["task_device"] == physical_card
            for key in (
                "kernel_config.py",
                "kernels/aic/qk_pv_aic.cpp",
                "kernels/aiv/qk_pv_aiv.cpp",
                "ptoas/qk_pv.cpp",
            ):
                assert _sha(raw("generated/" + variant + "-artifact/" + key)) == receipt["files"][key]
            # Busy counters overlap. Verify valid AIC/AIV rows without deriving wait.
            rows = list(csv.DictReader(io.StringIO(raw(prefix + "/data/pmu-2.csv").decode())))
            for function, core_type, count in ((3, 0, 24), (4, 1, 48)):
                values = [x for x in rows if int(x["func_id"]) == function and int(x["core_type"]) == core_type]
                assert len(values) == count and len({x["core_id"] for x in values}) == count
                assert all(int(x["pmu_total_cycles"]) > 0 and int(x["event_type"]) == 2 for x in values)


def test_live_staging_reproduction_profile_matches_actual_whole_runs():
    """The profile binds both completed whole-model runs, not just preparation."""
    m = _manifest()
    tool = _ROOT / "tests/pypto_test/prepare_tnd_precision_performance_repro.py"
    profile = runpy.run_path(str(tool))["PROFILES"]["swa_live"]
    assert profile["source_sha"] == m["source_manifest_sha256"]
    for run in m["whole_model"]:
        _artifact(run["archive"])
        with tarfile.open(_EVIDENCE / run["archive"]["path"]) as bundle:
            cfg = json.load(bundle.extractfile("results/configuration.json"))
        assert profile["helper_sha"] == cfg["helper_manifest_sha256"]
        assert profile["old_job"] == str(Path(cfg["snapshot"]).parents[1])
        assert profile["job_expression"] == profile["old_job"]
