"""Record exact Worker evidence for the 128K Score long tail; read-only."""
import collections
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
report = json.loads((RESULTS / "csa_split_optimization_20260927/baseline_2dd51f15/h131072_b16/swimlane/report.json").read_text())
path = Path(report["swimlane_windows"][0]["merged_swimlane"])
events = json.loads(path.read_text())["traceEvents"]
pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"}
workers = [e for e in events if e.get("pid") in pids and e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})]
origin = min(e["ts"] for e in workers)
names = {e["tid"]: e["args"]["name"] for e in events if e.get("pid") in pids and e.get("ph") == "M" and e.get("name") == "thread_name"}
score = [e for e in workers if "indexer_score_topk_native_pair_aic" in e["name"]]
counts = collections.Counter(names[e["tid"]] for e in score)
focus = []
for e in workers:
    if names[e["tid"]] in ("AIC_0", "AIC_3", "AIV_24", "AIV_25", "AIV_30", "AIV_31") and e["ts"]+e["dur"]-origin > 275 and e["ts"]-origin < 1060:
        focus.append({"worker": names[e["tid"]], "task": e["name"], "receive_us": e["ts"]-origin, "setup_us": e["args"]["local_setup_us"], "kernel_us": e["args"]["kernel-duration-us"], "end_us": e["ts"]+e["dur"]-origin})
out = {"source": str(path), "baseline": "2dd51f15", "scope": "one observed bad window, no frequency estimate", "score_aic_counts": dict(counts), "events": focus, "interpretation": "AIV_24/25 run merge before Score. AIC_0 receives no Score, AIC_3 receives two; test disabling Score producer early resolution, not merge producer."}
(ROOT / "baseline_tail_evidence.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out["score_aic_counts"]))
