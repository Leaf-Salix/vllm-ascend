"""Match separately collected level1 timing and dependencies with hard coverage gates."""

import collections
import hashlib
import json
import statistics
from pathlib import Path

r = Path("/Users/jiayetcs/Desktop/Project/PyPTO/reports/dsv4-nal-gap-20261026")
dep_root = r / "collected/lookahead-dep8k-v1"
timing_root = r / "collected/lookahead-8k-b32-v1"
report = json.loads((dep_root / "result.json").read_text())
timing = json.loads((timing_root / "result.json").read_text())
assert report["pass"] and report["complete"] and report["dependency_generation"] and not report["timing_executed"]
assert timing["pass"] and timing["complete"] and timing["timing_executed"] and not timing["performance"]["dep_gen"]
# Both runs validate immutable v7 helper and CPU/JIT kernels; additionally compare actual recorded source receipt.
assert report["derived_base_sha256"] == timing["helper_sha"]["probe_perf.py"]
assert report["manifest_sha256"] == timing["manifest_sha256"]
maps = {}
for i, window in enumerate(report["DFX"]["windows"]):
    path = dep_root / "dfx"
    if i:
        path = path / f"window_{i}"
    d = json.loads((path / "deps.json").read_text())
    mapping = {int(t["task_id"]): t["kernel_ids"] for t in d["tasks"]}
    arm = window["arm"]
    if arm in maps:
        assert maps[arm] == mapping, (arm, "dependency run instability")
    maps[arm] = mapping
rows = []
for i, window in enumerate(timing["performance"]["l1_windows"]):
    arm = window["arm"]
    if arm not in maps:
        continue
    path = timing_root / "dfx"
    if i:
        path = path / f"window_{i}"
    d = json.loads((path / "chip_swimlane_records.json").read_text())
    assert d["chip_swimlane_level"] == 1
    assert set(row[0] for row in d["aicore_tasks"]) == set(range(72))
    funcs = collections.defaultdict(list)
    counts = collections.Counter()
    tasks = set()
    for core, token, registration, start, end, _ in d["aicore_tasks"]:
        assert 0 < start <= end
        assert token in maps[arm], (arm, i, "timing token absent from dependency map", token)
        tasks.add(token)
        ids = maps[arm][token]
        assert len(set(k for k in ids[1:] if k >= 0)) <= 1
        fid = ids[0] if core < 24 else next(k for k in ids[1:] if k >= 0)
        name = report["kernel_maps"][arm]["names"][str(fid)]
        funcs[name].append((start, end))
        counts[(core, token, registration)] += 1
    assert all(v == 1 for v in counts.values())
    lo = min(t[3] for t in d["aicore_tasks"])
    freq = d["metadata"]["clock_freq_hz"]
    us = lambda ticks, frequency=freq: ticks * 1e6 / frequency
    rows.append(
        {
            "arm": arm,
            "repeat": window["repeat"],
            "window": i,
            "dependency_task_count": len(maps[arm]),
            "recorded_task_count": len(tasks),
            "all_recorded_tasks_covered": True,
            "span_us": us(max(t[4] for t in d["aicore_tasks"]) - lo),
            "functions": {
                name: {
                    "max_core_us": us(max(e - s for s, e in intervals)),
                    "inclusive_span_us": us(max(e for s, e in intervals) - min(s for s, e in intervals)),
                    "start_us": us(min(s for s, e in intervals) - lo),
                    "end_us": us(max(e for s, e in intervals) - lo),
                    "records": len(intervals),
                }
                for name, intervals in funcs.items()
            },
        }
    )
medians = {}
for arm in ("baseline", "candidate"):
    armrows = [v for v in rows if v["arm"] == arm]
    assert len(armrows) == 9
    assert all(set(v["functions"]) == set(armrows[0]["functions"]) for v in armrows)
    medians[arm] = {
        name: {
            field: statistics.median(v["functions"][name][field] for v in armrows)
            for field in ("max_core_us", "inclusive_span_us", "start_us", "end_us")
        }
        for name in armrows[0]["functions"]
    }
result = {
    "pass": True,
    "scope": (
        "L1 inclusive spans include Scalar, synchronization and transfer; "
        "overlapping functions are not additive; not pure compute split"
    ),
    "separate_timing_and_dependency": True,
    "windows": rows,
    "functions": medians,
    "timing_report_sha256": hashlib.sha256((timing_root / "result.json").read_bytes()).hexdigest(),
    "dependency_report_sha256": hashlib.sha256((dep_root / "result.json").read_bytes()).hexdigest(),
}
milestones = {}
for arm in ("baseline", "candidate"):
    samples = []
    for row in rows:
        if row["arm"] != arm:
            continue
        topk = row["functions"]["indexer_topk_single_leaf_publish"]["end_us"]
        heads = row["functions"]["native_heads"]["end_us"]
        end = row["span_us"]
        assert 0 < topk < heads < end
        samples.append(
            {
                "entry_to_topk_us": topk,
                "topk_to_heads_us": heads - topk,
                "heads_to_end_us": end - heads,
                "whole_us": end,
            }
        )
    milestones[arm] = {key: statistics.median(value[key] for value in samples) for key in samples[0]}
result["nonoverlap_milestones"] = milestones
(r / "LOOKAHEAD_STAGE_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
for arm, v in medians.items():
    print(arm)
    for name, value in sorted(v.items(), key=lambda item: -item[1]["inclusive_span_us"]):
        print(name, value)
