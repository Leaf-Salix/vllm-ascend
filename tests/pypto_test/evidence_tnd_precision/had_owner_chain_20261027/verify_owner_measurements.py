"""从原始归档复核HC owner链双向计时、实际注册见证和Native精度范围。"""

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path


def read_json(archive, suffix):
    members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
    assert len(members) == 1, (suffix, len(members))
    stream = archive.extractfile(members[0])
    assert stream is not None
    return json.load(stream)


def verify(path, accuracy_expected):
    with tarfile.open(path) as archive:
        d = read_json(archive, "/result.json")
        assert d["complete"] and d["pass"] and d["timing_executed"]
        assert d["per_run_registration_observed"]
        assert d["actual_capture_order"] == d["planned_capture_order"]
        assert d["true_tokens"] == d["physical_graph_tokens"] == 192
        assert d["history"] == 8192 and d["layer"] == 4
        assert d["per_request_padding"] == 0
        assert d["accuracy_executed"] is accuracy_expected
        identities = d["actual_execution_identity"]
        children = {}
        for arm in ("baseline", "candidate"):
            identity = identities[arm]
            for key in (
                "CPU_all_registered_child_bytes_equal",
                "orchestration_strict_alpha_equal",
                "compiled_meta_byte_equal",
                "kernel_config_AST_equal",
                "SDK_require_registration_checked_ready_PID_generation_owner",
            ):
                assert identity[key] is True, (arm, key)
            assert identity["registration"]["registered_callable_is_loaded_object"]
            children[arm] = identity["registered_children"]
            assert len(children[arm]) == 63
            assert all(c["CPU_serialized_child_byte_equal"] for c in children[arm].values())
        changed = [
            k
            for k, v in children["baseline"].items()
            if v["CPU_binary_sha256"] != children["candidate"][k]["CPU_binary_sha256"]
        ]
        assert changed == ["0", "1"]
        assert [children["candidate"][k]["name"] for k in changed] == [
            "packed_prefix_indexer_cube",
            "prepacked_prefix_indexer_vector",
        ]
        p = d["performance"]
        assert p["replays_per_event_window"] == 100 and not p["dep_gen"]
        assert len(d["execution_checks"]) == 44 and all(c["pass"] for c in d["execution_checks"])
        for row in p["l1_windows"]:
            suffix = "/dfx/" + row["raw"].split("/dfx/", 1)[1]
            raw = read_json(archive, suffix)
            records = raw["aicore_tasks"]
            meta = raw["metadata"]
            assert raw["chip_swimlane_level"] == 1 and meta["clock_freq_hz"] > 0
            assert meta["core_types"] == ["aic"] * 24 + ["aiv"] * 48
            assert meta["dropped_run_boundaries"] == 0 and len(meta["run_boundaries"]) == 1
            assert {r[0] for r in records} == set(range(72))
            assert all(len(r) == 6 and 0 < r[3] <= r[4] for r in records)
            span = (max(r[4] for r in records) - min(r[3] for r in records)) * 1e6 / meta["clock_freq_hz"]
            assert abs(span - row["us"]) < 1e-9
        summary = {}
        for arm in ("baseline", "candidate"):
            event = [v["us"] for v in p["event_windows"] if v["arm"] == arm]
            l1 = [v["us"] for v in p["l1_windows"] if v["arm"] == arm]
            assert len(event) == 12 and len(l1) == 9
            assert event == p["summary"][arm]["event_us"] and l1 == p["summary"][arm]["l1_us"]
            summary[arm] = {"event_us": statistics.median(event), "l1_us": statistics.median(l1)}
            assert summary[arm]["event_us"] == p["summary"][arm]["event_median_us"]
            assert summary[arm]["l1_us"] == p["summary"][arm]["l1_median_us"]
        gain = 1 - summary["candidate"]["event_us"] / summary["baseline"]["event_us"]
        l1_gain = 1 - summary["candidate"]["l1_us"] / summary["baseline"]["l1_us"]
        wins = sum(
            a < b
            for a, b in zip(p["summary"]["candidate"]["event_us"], p["summary"]["baseline"]["event_us"], strict=True)
        )
        assert p["candidate_vs_control"]["baseline"] == {
            "event_improvement": gain,
            "paired_event_wins": wins,
            "l1_improvement": l1_gain,
        }
        qualified = gain >= 0.005 and wins >= 10 and l1_gain > 0
        assert p["winners"] == (["candidate"] if qualified else [])
        if accuracy_expected:
            for arm in ("baseline", "candidate"):
                for phase in ("comparisons", "B_comparisons"):
                    assert len(d[phase][arm]) == 8
                    assert all(m["bit_equal"] for m in d[phase][arm].values())
            assert all(d["graphs"][arm]["live_input_ABA"] for arm in ("native", "baseline", "candidate"))
        return {
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": summary,
            "event_improvement": gain,
            "l1_improvement": l1_gain,
            "paired_wins": wins,
            "performance_pass": qualified,
            "accuracy_executed": accuracy_expected,
            "Native_A_B_8_ABA_bit_equal": True if accuracy_expected else None,
            "actual_registration_63_checked": True,
            "only_changed_func": [0, 1],
            "guard_readonly_checks": 44,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        name: verify(args.evidence_dir / name, accuracy)
        for name, accuracy in (
            ("hc-owner-first-v2-evidence.tar.gz", False),
            ("hc-owner-reverse-v2-evidence.tar.gz", True),
        )
    }
    args.out.write_text(json.dumps({"pass": True, "runs": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
