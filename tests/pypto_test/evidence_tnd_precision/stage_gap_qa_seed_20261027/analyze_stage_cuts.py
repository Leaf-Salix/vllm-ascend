"""Join separate L1/dependency runs; fail on any identity or task coverage gap."""

import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEP = ROOT / "collected/direct-dependency/results/direct-dependency-8k-b32-v1"
TIMING = ROOT / "collected/cumulative-dfx/results/nal-cumulative-cuts-8k-b32-dfx-v1"
dep = json.loads((DEP / "result.json").read_text())
timing = json.loads((TIMING / "result.json").read_text())
assert dep["pass"] and dep["complete"] and dep["dependency_generation"] and not dep["timing_executed"]
assert timing["pass"] and timing["complete"] and not timing["DFX"]["enable_dep_gen"]
assert dep["derived_parent_sha256"] == timing["helper_sha"]["probe_common.py"]
for key in ("manifest_sha256", "source_manifest_sha256", "runtime77", "qa_sha256", "tested_sources"):
    assert dep[key] == timing[key], key
for key in ("source_heads", "binary_sha256", "ptoas_binary_sha256"):
    assert dep["our_reused_compile_proof"]["sdk"][key] == timing["our_reused_compile_proof"]["sdk"][key], key
actual_dir = ROOT / "collected/cumulative-jit-map/pypto-build-deg81an5"
actual = json.loads((actual_dir / "binary_manifest.json").read_text())
actual_names = {str(k["func_id"]): k["name"] for k in actual["kernels"]}
assert actual_names == dep["kernel_maps"]["candidate"]["names"]
assert {str(k["func_id"]): k["binary"]["sha256"] for k in actual["kernels"]} == dep["kernel_maps"]["candidate"][
    "binary_sha256"
]
assert dep["kernel_maps"]["candidate"]["orchestration_proof"]["alpha_equivalent"]
names = dep["kernel_maps"]["candidate"]["names"]
mappings = []
for i, _ in enumerate(dep["DFX"]["windows"]):
    p = DEP / "dfx"
    if i:
        p /= f"window_{i}"
    d = json.loads((p / "deps.json").read_text())
    mappings.append({int(t["task_id"]): t["kernel_ids"] for t in d["tasks"]})
assert len(mappings) == 2 and mappings[0] == mappings[1], "unstable dependency map"
rows = []
for i, window in enumerate(timing["DFX"]["windows"]):
    if window["arm"] != "candidate":
        continue
    p = TIMING / "dfx"
    if i:
        p /= f"window_{i}"
    d = json.loads((p / "chip_swimlane_records.json").read_text())
    assert d["chip_swimlane_level"] == 1
    assert d["metadata"]["dropped_run_boundaries"] == 0
    assert len(d["metadata"]["run_boundaries"]) == 1
    assert {x[0] for x in d["aicore_tasks"]} == set(range(72))
    funcs = collections.defaultdict(list)
    unique = collections.Counter()
    tokens = set()
    for core, token, registration, start, end, _ in d["aicore_tasks"]:
        assert 0 < start <= end
        assert token in mappings[0], ("missing task token", i, token)
        ids = mappings[0][token]
        assert len({k for k in ids[1:] if k >= 0}) <= 1
        fid = ids[0] if core < 24 else next(k for k in ids[1:] if k >= 0)
        funcs[names[str(fid)]].append((start, end))
        unique[core, token, registration] += 1
        tokens.add(token)
    assert all(v == 1 for v in unique.values())
    low = min(x[3] for x in d["aicore_tasks"])
    frequency = d["metadata"]["clock_freq_hz"]

    def us(ticks, frequency=frequency):
        return ticks * 1e6 / frequency

    row = {
        "repeat": window["repeat"],
        "recorded_task_count": len(tokens),
        "all_recorded_tasks_covered": True,
        "whole_us": us(max(x[4] for x in d["aicore_tasks"]) - low),
        "functions": {
            name: {
                "start_us": us(min(s for s, e in samples) - low),
                "end_us": us(max(e for s, e in samples) - low),
                "inclusive_span_us": us(max(e for s, e in samples) - min(s for s, e in samples)),
                "max_core_us": us(max(e - s for s, e in samples)),
                "records": len(samples),
            }
            for name, samples in funcs.items()
        },
    }
    topk = row["functions"]["indexer_topk_single_leaf_publish"]["end_us"]
    heads = row["functions"]["qk_pv_aiv"]["end_us"]
    assert 0 < topk < heads < row["whole_us"]
    plan = row["functions"]["csa_slots_build_valid_qk_plan"]["end_us"]
    assert topk < plan < heads
    row["milestones"] = {
        "entry_to_topk_us": topk,
        "topk_to_heads_us": heads - topk,
        "entry_to_topk_plan_us": plan,
        "plan_to_heads_us": heads - plan,
        "heads_to_end_us": row["whole_us"] - heads,
    }
    rows.append(row)
assert len(rows) == 8
assert all(set(x["functions"]) == set(rows[0]["functions"]) for x in rows)
medians = {key: statistics.median(x["milestones"][key] for x in rows) for key in rows[0]["milestones"]}
functions = {
    name: {
        field: statistics.median(x["functions"][name][field] for x in rows)
        for field in ("start_us", "end_us", "inclusive_span_us", "max_core_us")
    }
    for name in rows[0]["functions"]
}
result = {
    "pass": True,
    "scope": (
        "same-source/new-SDK separate dep and L1; named inclusive spans include transfer/synchronization; "
        "overlapping functions not additive; Nal cut differences are diagnostic only"
    ),
    "actual_l1_kernel_names_and_binary_sha_match_dependency": True,
    "dependency_task_count": len(mappings[0]),
    "windows": rows,
    "milestones_medians": medians,
    "functions_medians": functions,
    "timing_result_sha256": hashlib.sha256((TIMING / "result.json").read_bytes()).hexdigest(),
    "dependency_result_sha256": hashlib.sha256((DEP / "result.json").read_bytes()).hexdigest(),
}
(ROOT / "NEW_SDK_STAGE_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
print("milestones", medians)
for name, value in sorted(functions.items(), key=lambda item: -item[1]["inclusive_span_us"]):
    print(name, value)
