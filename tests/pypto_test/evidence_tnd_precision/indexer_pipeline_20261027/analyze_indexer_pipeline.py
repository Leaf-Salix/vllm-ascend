"""Recompute full timing and join separate dependency maps, without NPU access."""

import collections
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMING = HERE / "indexer-first-raw.tgz"
DEPENDENCY = HERE / "indexer-dependency-raw.tgz"

with tarfile.open(TIMING) as timing_tar, tarfile.open(DEPENDENCY) as dep_tar:

    def load(archive, path):
        return json.load(archive.extractfile(path))

    timing = load(timing_tar, "indexer-chain-8k32-first/result.json")
    dependency = load(dep_tar, "indexer-chain-dependency/result.json")
    assert timing["pass"] and timing["complete"] and timing["timing_executed"]
    assert dependency["pass"] and dependency["complete"] and dependency["dependency_generation"]
    assert dependency["timing_executed"] is False
    for record in (timing, dependency):
        assert record["true_tokens"] == record["physical_graph_tokens"] == 192
        assert record["lengths"] == [6] * 32
        assert record["history"] == 8192 and record["layer"] == 4
        assert record["per_request_padding"] == 0
        assert record["accuracy_executed"] is False
        assert record["actual_capture_order"] == record["planned_capture_order"]
        assert record["per_run_registration_observed"]
    assert len(timing["execution_checks"]) == 44
    assert all(check["pass"] for check in timing["execution_checks"])
    assert dependency["derived_parent_sha256"] == timing["helper_sha"]["probe_perf.py"]
    for key in ("helper_sha", "source_manifest_sha256", "runtime77", "qa_sha256", "tested_sources"):
        assert dependency[key] == timing[key], key
    for key in ("source_heads", "binary_sha256", "ptoas_binary_sha256"):
        assert dependency["sdk"][key] == timing["sdk"][key], key
    maps = {}
    names = {}
    for arm in ("baseline", "candidate"):
        old = timing["actual_execution_identity"][arm]
        new = dependency["actual_execution_identity"][arm]
        for key in (
            "registered_children",
            "normalized_orchestration_sha256",
            "compiled_meta_byte_equal",
            "kernel_config_AST_equal",
        ):
            assert old[key] == new[key], (arm, key)
        assert old["CPU_all_registered_child_bytes_equal"] and new["CPU_all_registered_child_bytes_equal"]
        names[arm] = {int(k): v["name"] for k, v in old["registered_children"].items()}
        selected = [w for w in dependency["DFX"]["windows"] if w["arm"] == arm]
        assert len(selected) == 2
        for window in selected:
            path = "indexer-chain-dependency/dfx/" + window["path"].split("/dfx/", 1)[1]
            raw = dep_tar.extractfile(path).read()
            assert hashlib.sha256(raw).hexdigest() == window["sha256"]
            data = json.loads(raw)
            mapping = {int(t["task_id"]): t["kernel_ids"] for t in data["tasks"]}
            if arm in maps:
                assert maps[arm] == mapping, (arm, "unstable task IDs")
            maps[arm] = mapping

    windows = []
    for window in timing["performance"]["l1_windows"]:
        arm = window["arm"]
        path = "indexer-chain-8k32-first/dfx/" + window["raw"].split("/dfx/", 1)[1]
        data = load(timing_tar, path)
        assert data["chip_swimlane_level"] == 1
        assert data["metadata"]["dropped_run_boundaries"] == 0
        assert len(data["metadata"]["run_boundaries"]) == 1
        records = data["aicore_tasks"]
        assert {r[0] for r in records} == set(range(72))
        assert len({tuple(r[:3]) for r in records}) == len(records)
        origin = min(r[3] for r in records)
        hz = data["metadata"]["clock_freq_hz"]
        assert hz > 0
        us = lambda ticks, frequency=hz: ticks * 1e6 / frequency
        full = us(max(r[4] for r in records) - origin)
        assert math.isclose(full, window["us"], abs_tol=1e-8)
        functions = collections.defaultdict(list)
        for core, token, _registration, start, end, _ in records:
            assert 0 < start <= end and token in maps[arm], (arm, token)
            ids = maps[arm][token]
            vectors = {i for i in ids[1:] if i >= 0}
            assert len(vectors) <= 1
            fid = ids[0] if core < 24 else next(iter(vectors))
            functions[names[arm][fid]].append((start, end))
        scheduling = None
        if arm == "candidate":
            mixed_names = [n for n in functions if n.startswith("complete_indexer_pipeline") and n.endswith("_aic")]
            assert len(mixed_names) == 1
            mixed = functions[mixed_names[0]]
            qb = functions["qproj_matmul"]
            assert len(mixed) == len(qb) == 24
            begin, finish = min(s for s, e in mixed), max(e for s, e in mixed)
            before = sum(e <= begin for s, e in qb)
            after = sum(s >= finish for s, e in qb)
            overlap = len(qb) - before - after
            scheduling = {
                "qb_blocks_before_mix": before,
                "qb_blocks_after_mix": after,
                "qb_blocks_overlap_mix_envelope": overlap,
                "mix_start_us": us(begin - origin),
                "mix_end_us": us(finish - origin),
                "qb_last_end_us": us(max(e for s, e in qb) - origin),
            }
        by_function = {
            name: {
                "start_us": us(min(s for s, e in samples) - origin),
                "end_us": us(max(e for s, e in samples) - origin),
                "inclusive_span_us": us(max(e for s, e in samples) - min(s for s, e in samples)),
                "max_core_us": us(max(e - s for s, e in samples)),
            }
            for name, samples in functions.items()
        }
        readiness = {
            "q_ready_us": by_function["qproj_dequant_rms_nope_rope"]["end_us"],
            "plan_ready_us": by_function["csa_slots_build_valid_qk_plan"]["end_us"],
            "attention_start_us": by_function["qk_pv_aiv"]["start_us"],
        }
        readiness["later_required_input"] = "q" if readiness["q_ready_us"] > readiness["plan_ready_us"] else "plan"
        windows.append(
            {
                "arm": arm,
                "repeat": window["repeat"],
                "whole_us": full,
                "all_records_identity_mapped": True,
                "functions": by_function,
                "qb_scheduling": scheduling,
                "readiness": readiness,
            }
        )

    stages = {}
    measurements = {}
    for arm in ("baseline", "candidate"):
        selected = [w for w in windows if w["arm"] == arm]
        assert len(selected) == 9
        function_set = set(selected[0]["functions"])
        assert all(set(w["functions"]) == function_set for w in selected)
        stages[arm] = {
            name: {
                field: statistics.median(w["functions"][name][field] for w in selected)
                for field in ("start_us", "end_us", "inclusive_span_us", "max_core_us")
            }
            for name in sorted(function_set)
        }
        event = [w["us"] for w in timing["performance"]["event_windows"] if w["arm"] == arm]
        assert len(event) == 12
        measurements[arm] = {
            "event_median_us": statistics.median(event),
            "l1_median_us": statistics.median(w["whole_us"] for w in selected),
        }
        assert measurements[arm]["event_median_us"] == timing["performance"]["summary"][arm]["event_median_us"]
        assert measurements[arm]["l1_median_us"] == timing["performance"]["summary"][arm]["l1_median_us"]
    wins = sum(
        c["us"] < b["us"]
        for c, b in zip(
            [w for w in timing["performance"]["event_windows"] if w["arm"] == "candidate"],
            [w for w in timing["performance"]["event_windows"] if w["arm"] == "baseline"],
            strict=True,
        )
    )
    result = {
        "pass": True,
        "measurements": measurements,
        "paired_event_wins": wins,
        "source_and_actual_registered_binary_maps_match": True,
        "accuracy_executed": timing["accuracy_executed"],
        "adopt": False,
        "functions_medians": stages,
        "windows": windows,
        "scope": (
            "Same-source separate L1/dependency runs; "
            "inclusive overlapping spans are not additive or pure compute costs"
        ),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (TIMING, DEPENDENCY)},
    }
    (HERE / "INDEXER_PIPELINE_STAGE_RECHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(measurements, indent=2))
    for arm in stages:
        print(arm)
        for name, values in stages[arm].items():
            if any(key in name for key in ("indexer", "qproj", "qk_pv", "qr_", "slots_build", "oproj")):
                print(name, values)
