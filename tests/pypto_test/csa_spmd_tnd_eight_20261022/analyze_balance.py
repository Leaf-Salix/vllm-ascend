# SPDX-License-Identifier: Apache-2.0
"""Read existing ten-window traces to separate task tails from whole-SPMD tails.

Elapsed intervals include waits. MPMD phase intervals locate hypotheses but do
not measure the internal phases of the separate SPMD implementation.
"""

import importlib.util
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def load_analysis(here):
    path = here.parent / "csa_mpmd_tnd_20261021/analyze_task_l1.py"
    spec = importlib.util.spec_from_file_location("existing_csa_l1_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spmd_window(window):
    raw = read(window["raw"])
    rows = raw["aicore_tasks"]
    meta = raw["metadata"]
    start = min(r[3] for r in rows)
    factor = 1e6 / meta["clock_freq_hz"]
    roles = [
        dict(
            core=r[0],
            kind=meta["core_types"][r[0]],
            begin_us=(r[3] - start) * factor,
            end_us=(r[4] - start) * factor,
            elapsed_us=(r[4] - r[3]) * factor,
        )
        for r in rows
    ]
    summary = {}
    for kind in ("aic", "aiv"):
        selected = [r for r in roles if r["kind"] == kind]
        ends = [r["end_us"] for r in selected]
        spans = [r["elapsed_us"] for r in selected]
        summary[kind] = dict(
            earliest_end_us=min(ends),
            latest_end_us=max(ends),
            end_spread_us=max(ends) - min(ends),
            median_elapsed_us=statistics.median(spans),
            max_elapsed_us=max(spans),
        )
    return dict(raw=window["raw"], roles=roles, kinds=summary)


def analyze_case(frozen_case, helper):
    result_path = Path(frozen_case["source_report"])
    result = read(result_path)
    mpmd_windows = [w for w in result["windows"] if w["side"] == "mpmd"]
    counts = {w["task_groups"] for w in mpmd_windows}
    assert len(counts) == 1
    sources = result_path.parent.glob("build_output/decode_csa_tnd_*/orchestration/decode_csa_tnd.cpp")
    matches = [(str(p), helper.task_names(p)) for p in sources]
    matches = [(p, names) for p, names in matches if len(names) in counts]
    assert matches and all(names == matches[0][1] for _, names in matches)
    records = [helper.summarize_window(window, matches[0][1]) for window in mpmd_windows]
    stages = []
    for ordinal, name in matches[0][1].items():
        entries = [w["stages"][ordinal] for w in records]
        for kind in entries[0]["kinds"]:
            samples = [r["kinds"][kind] for r in entries]
            max_mean = [s["max_us"] / s["mean_us"] for s in samples]
            stages.append(
                dict(
                    name=name,
                    kind=kind,
                    ordinal=ordinal,
                    windows=len(samples),
                    median_instances=statistics.median(s["instances"] for s in samples),
                    median_max_us=statistics.median(s["max_us"] for s in samples),
                    median_mean_us=statistics.median(s["mean_us"] for s in samples),
                    median_max_over_mean=statistics.median(max_mean),
                    per_window_max_over_mean=max_mean,
                )
            )
    spmd = [spmd_window(w) for w in result["windows"] if w["side"] == "spmd"]
    return dict(
        case=result["case"],
        mpmd_orchestration_sources=[p for p, _ in matches],
        mpmd_windows=[dict(raw=w["raw"], span_us=w["span_us"]) for w in records],
        mpmd_stage_elapsed_balance=stages,
        spmd_windows=[dict(raw=w["raw"], kinds=w["kinds"]) for w in spmd],
        spmd_tail_summary={
            k: {name: statistics.median(w["kinds"][k][name] for w in spmd) for name in spmd[0]["kinds"][k]}
            for k in ("aic", "aiv")
        },
    )


def main():
    here = Path(__file__).resolve().parent
    baseline = read(here / "baseline.json")
    helper = load_analysis(here)
    cases = {key: analyze_case(value, helper) for key, value in baseline["cases"].items()}
    output = dict(
        source="baseline.json",
        device_execution=False,
        scope="Existing same-invocation raw L1 records; no new performance sampling",
        limitations=[
            "Elapsed time includes waits; max/mean is not arithmetic or pipe utilization",
            "A small active subset is expected for some kernels; inactive roles are not discarded",
            "MPMD tasks and SPMD phases have different schedules; do not transplant phase durations",
            "Cross-window medians do not form one execution timeline",
        ],
        cases=cases,
    )
    with (here / "baseline_balance.json").open("x") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for key, data in cases.items():
        selected = sorted(data["mpmd_stage_elapsed_balance"], key=lambda r: r["median_max_us"], reverse=True)[:4]
        print(key, data["case"]["name"])
        for stage in selected:
            print(
                f"  {stage['name']} / {stage['kind']}: max {stage['median_max_us']:.2f} us, "
                f"max/mean {stage['median_max_over_mean']:.2f}"
            )


if __name__ == "__main__":
    main()
