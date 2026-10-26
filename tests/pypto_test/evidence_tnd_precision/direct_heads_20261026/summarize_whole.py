"""复核本轮整模型原始证据，按对齐步骤取16rank最大耗时。"""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def distribution(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
        "max": max(values),
    }


def geometry(rows):
    # Async CPU input_ids staging is not the authoritative GPU token stream.
    excluded = {"call", "input_ids_cpu", "actual_graph_routes"}
    return [{k: v for k, v in row.items() if k not in excluded} for row in rows]


root = Path(sys.argv[1])
audit = json.loads((root / "audit.json").read_text())
configuration = json.loads((root / "configuration.json").read_text())
assert audit["pass"] and audit["complete"] and not list(root.glob("*STOP.json"))
assert configuration["pass"] and configuration["native_source_equal"]
batch = configuration["batch"]
assert audit["tokens_compared"] == 16 * batch * 192 * 2
raws, measurements, ranks = [], [], []
for rank, audited in enumerate(audit["ranks"]):
    path = root / f"rank{rank}.real_perf.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert audited["rank"] == rank and audited["helper_output_sha256"] == digest
    assert audited["tokens_equal"] and audited["events_equal"] and audited["preemptions"] == 0
    raw = json.loads(path.read_text())
    assert len(raw["rounds"]) == 1 and raw["warmup"]["decode_tokens"] == 192
    for runtime in raw["worker_runtime_config"]:
        assert runtime["deterministic_level"] == 1 and runtime["torch_deterministic"]
        assert runtime["hccl_deterministic"] == "true" and not runtime["dynamic_eplb"]
        assert runtime["cann_event_work_mode"] == 1
    for worker in raw["implementations"]:
        csa = [v for v in worker if v["kind"] == "csa"]
        hca = [v for v in worker if v["kind"] == "hca"]
        assert len(csa) == 21 and len(hca) == 20
        assert all(v["implementation"] == "pto" for v in csa)
        assert all(v["implementation"] == "native" for v in hca)
    round_ = raw["rounds"][0]
    measurement = round_["timing"][0]
    assert measurement == audited["timing"][0]
    assert len(measurement["forward"]["samples_us"]) == 20
    assert len(measurement["cycle"]["samples_us"]) == 19
    before, after = round_["identity_before"][0], measurement["identity"]
    for field in (
        "generation_builds",
        "binary_builds",
        "bypasses",
        "invalid_entries",
        "storage_errors",
        "forced_rebuilds",
    ):
        assert before["cache_stats"][field] == after["cache_stats"][field]
    if raws:
        assert geometry(measurement["metadata"]) == geometry(measurements[0]["metadata"])
        assert round_["requests"] == raws[0]["rounds"][0]["requests"]
        assert raw["warmup"]["requests"] == raws[0]["warmup"]["requests"]
    ranks.append(
        {
            "rank": rank,
            "raw_sha256": digest,
            "warm_tokens_events_sha256": hashlib.sha256(
                json.dumps(raw["warmup"]["requests"], sort_keys=True).encode()
            ).hexdigest(),
            "measured_tokens_events_sha256": hashlib.sha256(
                json.dumps(round_["requests"], sort_keys=True).encode()
            ).hexdigest(),
        }
    )
    raws.append(raw)
    measurements.append(measurement)
assert len(ranks) == 16
forward = [max(m["forward"]["samples_us"][i] for m in measurements) for i in range(20)]
cycle = [max(m["cycle"]["samples_us"][i] for m in measurements) for i in range(19)]
report = {
    "pass": True,
    "case": configuration["case"],
    "submitted_per_rank": batch,
    "measured_active_per_rank": measurements[0]["metadata"][0]["requests"],
    "real_tokens": measurements[0]["metadata"][0]["actual_tokens"],
    "physical_graph_bucket": measurements[0]["metadata"][0]["padded_tokens"],
    "tokens_compared": audit["tokens_compared"],
    "native_tokens_events_equal": True,
    "cross_rank_geometry_outputs_equal": True,
    "preemptions": 0,
    "forward_max_rank_us": distribution(forward),
    "cycle_max_rank_us": distribution(cycle),
    "ranks": ranks,
    "raw_rank0_full_outputs": raws[0]["rounds"][0]["requests"],
    "audit_script_sha256": audit["audit_script_sha256"],
    "audit_sha256": hashlib.sha256((root / "audit.json").read_bytes()).hexdigest(),
    "source_manifest_sha256": configuration["source"]["source_manifest_sha256"],
    "candidate": configuration["candidate"],
    "scope": (
        "One measured 192-token round after warm192. Per aligned step maximum over 16 ranks; CPU geometry matched. "
        "No fresh Native/frozen whole timing arm; not a paired whole-model speedup. "
        "Cycle includes sampling/draft/scheduling and CPU metadata observer; not pure CSA."
    ),
}
(root / "SUMMARY.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(
    report["case"],
    report["tokens_compared"],
    "Native-exact tokens; forward mean ms",
    statistics.mean(forward) / 1000,
    "cycle mean ms",
    statistics.mean(cycle) / 1000,
)
