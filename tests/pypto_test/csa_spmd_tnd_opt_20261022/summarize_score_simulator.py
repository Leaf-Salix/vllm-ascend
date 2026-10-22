# SPDX-License-Identifier: Apache-2.0
"""Summarize isolated producer traces without treating overlap as utilization."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def merged_intervals(events):
    intervals = []
    for start, end in sorted((e["ts"], e["ts"] + e["dur"]) for e in events):
        if not intervals or start > intervals[-1][1]:
            intervals.append([start, end])
        else:
            intervals[-1][1] = max(end, intervals[-1][1])
    return intervals


def interval_summary(events):
    intervals = merged_intervals(events)
    assert intervals
    return {
        "instruction_count": len(events),
        "names": dict(collections.Counter(e["name"] for e in events)),
        "first_us": intervals[0][0],
        "last_us": intervals[-1][1],
        "span_us": intervals[-1][1] - intervals[0][0],
        "visible_interval_union_us": sum(end - start for start, end in intervals),
        "visible_interval_sum_us": sum(e["dur"] for e in events),
        "largest_gap_us": max((b[0] - a[1] for a, b in zip(intervals, intervals[1:])), default=0),
    }


def summarize(job):
    result = json.loads((job / "result.json").read_text())
    assert result["passed"] and result["output_verified"]
    assert result["formal_gate_eligible"] is False
    manifest = json.loads((job / "job_manifest.json").read_text())
    for relative, expected in manifest["immutable_files"].items():
        assert hashlib.sha256((job / relative).read_bytes()).hexdigest() == expected, relative
    trace_path = Path(result["collection"]["trace_json"])
    trace = json.loads(trace_path.read_text())
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    csv_path = next(trace_path.parent.rglob("*instr_exe.csv"))
    with csv_path.open() as stream:
        instruction_rows = list(csv.DictReader(stream))
    mmads = [e for e in events if e["name"] == "MMAD"]
    csv_mmads = [r for r in instruction_rows if r["instr"] == "MMAD"]
    assert len(mmads) == sum(int(r["call_count"]) for r in csv_mmads)
    # Chrome timestamps are us; displayTimeUnit only controls the UI label.
    # The simulator's MMAD durations independently agree with its cycle CSV at
    # 1850 cycles/us. Do not extend that equality to DMA/flag queue lifetimes.
    clock_mhz = sum(int(r["cycles"]) for r in csv_mmads) / sum(e["dur"] for e in mmads)
    assert abs(clock_mhz - 1850) < 0.01, clock_mhz
    pipes = {
        pipe: interval_summary([e for e in events if e["tid"] == pipe])
        for pipe in ("MTE2", "MTE1", "CUBE", "FIXP", "SCALAR")
    }
    pipes["CUBE_MMAD_ONLY"] = interval_summary(mmads)
    for dtype in ("S8", "F16"):
        selected = [e for e in mmads if "dtype:" + dtype in e["args"]["detail"]]
        pipes["MMAD_" + dtype] = interval_summary(selected)
    mte2_bytes = sum(int(e["args"].get("process_bytes", 0)) for e in events if e["tid"] == "MTE2")
    return {
        "job": str(job),
        "label": manifest["label"],
        "model": result["simulator_soc"],
        "shared_toolchain": result["shared_toolchain"],
        "output_verified": True,
        "output_check": result["output_check"],
        "trace": str(trace_path),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "instruction_csv": str(csv_path),
        "instruction_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "immutable_files": manifest["immutable_files"],
        "timestamp_unit": "us",
        "display_time_unit": trace["displayTimeUnit"],
        "mmad_cycle_crosscheck_mhz": clock_mhz,
        "complete_event_envelope": interval_summary(events),
        "mte2_bytes": mte2_bytes,
        "pipes": pipes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = [summarize(job.resolve()) for job in args.jobs]
    report = {
        "scope": "synthetic unthrottled single-AIC Score producer; six 1024-column steps",
        "formal_gate_eligible": False,
        "limitations": [
            "No FIFO consumer, inter-core schedule, real cache aliasing, or hardware latency prediction",
            "Trace intervals are instruction lifetimes; union is not physical pipe utilization",
            "Overlapping instruction durations must not be summed as elapsed time",
            "CSV non-MMAD cycle sums are not interchangeable with trace DMA/flag lifetimes",
            "Synthetic relative spans cannot enter the nine-case L1 acceptance gate",
        ],
        "cases": cases,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for case in cases:
        print(
            case["label"],
            {k: case["complete_event_envelope"][k] for k in ("span_us", "first_us", "last_us")},
            "MMAD", case["pipes"]["CUBE_MMAD_ONLY"]["instruction_count"],
            "MTE2 bytes", case["mte2_bytes"],
        )


if __name__ == "__main__":
    main()
