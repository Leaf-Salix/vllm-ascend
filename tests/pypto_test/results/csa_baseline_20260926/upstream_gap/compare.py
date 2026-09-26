"""CPU-only comparison of the retained integration and historical upstream traces."""

import collections
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALIASES = {
    "proj_a_mm_0": "proj_a_mm", "proj_b_mm_0": "proj_b_mm",
    "_proj_b_mm_nz_kernel": "proj_b_mm", "quant_0": "quant", "proj_b_act_0": "proj_b_act",
}


def name(event):
    value = re.sub(r"\((?:r\d+)?t\d+\)$", "", event["name"]).removesuffix("_spmd")
    return ALIASES.get(value, value)


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def summarize(path):
    data = json.loads(path.read_text())
    events = data["traceEvents"]
    processes = {e["pid"]: e["args"]["name"] for e in events
                 if e.get("ph") == "M" and e.get("name") == "process_name"}
    core_names = {(e["pid"], e["tid"]): e["args"]["name"] for e in events
                  if e.get("ph") == "M" and e.get("name") == "thread_name"}
    workers = [e for e in events if e.get("ph") == "X" and
               processes.get(e.get("pid")) == "Worker View" and
               "kernel-duration-us" in e.get("args", {})]
    scheduler = [e for e in events if e.get("ph") == "X" and
                 processes.get(e.get("pid")) == "Scheduler View" and
                 "dispatch-time-us" in e.get("args", {})]
    # SPMD can reuse a physical core for the same logical task. Match occurrences,
    # not just taskId/core; otherwise later records silently overwrite earlier ones.
    worker_keys, scheduler_keys = collections.defaultdict(list), collections.defaultdict(list)
    for rows, buckets in ((workers, worker_keys), (scheduler, scheduler_keys)):
        for event in rows:
            buckets[event["args"]["event-hint"]].append(event)
    joined = {}
    for key, rows in worker_keys.items():
        other = scheduler_keys.get(key, [])
        if not other:
            continue
        if len(rows) != len(other):
            raise ValueError(f"incomplete scheduler records: {key}")
        for worker, sched in zip(sorted(rows, key=lambda e: e["ts"]), sorted(other, key=lambda e: e["ts"])):
            joined[worker["id"]] = sched
    begin = min(e["ts"] for e in workers)
    end = max(e["ts"] + e["dur"] for e in workers)
    groups = collections.defaultdict(list)
    for event in workers:
        groups[name(event)].append(event)
    kernels = {}
    for label, rows in groups.items():
        times = [e["args"]["kernel-duration-us"] for e in rows]
        setups = [e["args"]["local_setup_us"] for e in rows]
        first = min(e["ts"] for e in rows)
        last = max(e["ts"] + e["dur"] for e in rows)
        cores = collections.Counter(e["tid"] for e in rows)
        values = {"count": len(rows), "physical_cores": len(cores), "max_instances_per_core": max(cores.values()),
                  "kernel_mean_us": statistics.mean(times), "kernel_max_us": max(times),
                  "kernel_core_us": sum(times), "setup_mean_us": statistics.mean(setups),
                  "setup_core_us": sum(setups), "first_us": first - begin, "last_us": last - begin,
                  "span_us": last - first, "receive_spread_us": max(e["ts"] for e in rows) - first}
        pairs = [(e, joined[e["id"]]) for e in rows if e["id"] in joined]
        if pairs:
            values["scheduler"] = {
                "samples": len(pairs),
                "dispatch_spread_us": max(s["ts"] for _, s in pairs) - min(s["ts"] for _, s in pairs),
                "dispatch_to_receive_mean_us": statistics.mean(w["ts"] - s["ts"] for w, s in pairs),
                "end_to_finish_mean_us": statistics.mean(s["ts"] + s["dur"] - w["ts"] - w["dur"]
                                                          for w, s in pairs),
            }
        kernels[label] = values
    occupancy = {}
    for kind, count in (("AIC", 24), ("AIV", 48)):
        rows = [e for e in workers if core_names[e["pid"], e["tid"]].startswith(kind + "_")]
        kernel = sum(e["args"]["kernel-duration-us"] for e in rows)
        setup = sum(e["args"]["local_setup_us"] for e in rows)
        capacity = count * (end - begin)
        occupancy[kind] = {"kernel_core_us": kernel, "setup_core_us": setup,
                           "kernel_percent": kernel / capacity * 100,
                           "setup_percent": setup / capacity * 100,
                           "outside_worker_percent": (capacity - kernel - setup) / capacity * 100}
    return {"path": str(path), "metadata": data.get("metadata", {}), "worker_count": len(workers),
            "worker_first_original_us": begin, "worker_last_original_us": end,
            "worker_span_us": end - begin, "scheduler_records": len(scheduler),
            "occupancy": occupancy, "kernels": kernels}


def main():
    report = {"scope": "Historical upstream Worker View versus retained integration; not matched acceptance.",
              "upstream": summarize(ROOT / "upstream_worker_trace.json"),
              "current": summarize(ROOT.parent / "perf_qproj_upstream/swimlane/dfx/merged_swimlane.json")}
    report = rounded(report)
    (ROOT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    rows = ["| Task | 数量 上游→当前 | 平均核内 μs 上游→当前 | 平均 setup μs 上游→当前 | 整组窗口 μs 上游→当前 |",
            "| --- | --- | --- | --- | --- |"]
    for key in dict.fromkeys([*report["upstream"]["kernels"], *report["current"]["kernels"]]):
        up = report["upstream"]["kernels"].get(key, {})
        current = report["current"]["kernels"].get(key, {})
        columns = []
        for field in ("count", "kernel_mean_us", "setup_mean_us", "span_us"):
            values = [str(v[field]) if field == "count" else f"{v[field]:.2f}" if field in v else "—"
                      for v in (up, current)] if field != "count" else [str(v.get(field, 0)) for v in (up, current)]
            columns.append(" → ".join(values))
        rows.append("| " + " | ".join([key, *columns]) + " |")
    (ROOT / "tasks.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({side: {k: v for k, v in report[side].items() if k != "kernels"}
                      for side in ("upstream", "current")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
