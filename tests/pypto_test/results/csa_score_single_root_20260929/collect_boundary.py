"""检查长S6的2048段尾部、六leaf单根和固定图padding。"""

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402

STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
          "indexer.0", "indexer.1", "indexer_state.0"}


def compact_padding(padding, source):
    """Keep coverage and outcomes; leave repeated tensor descriptions in raw reports."""
    replays = []
    for replay in padding["replays"]:
        checks = replay["state_comparison"]
        if set(checks) != STATES:
            raise ValueError("Padding state coverage incomplete")
        item = {"active_batch": replay["active_batch"], "reference": replay["reference"]}
        for name in ("state_comparison", "compact_metadata", "guards"):
            values = replay[name]
            failures = {k: v for k, v in values.items() if v["status"] != "PASS"}
            if failures:
                raise ValueError(f"Padding {name} failed: {failures}")
            item[name] = {"status": "PASS", "count": len(values), "names": sorted(values)}
        replays.append(item)
    return {"status": padding["status"], "bucket_batch": padding["bucket_batch"],
            "scope": padding["scope"], "source": str(source), "replays": replays}


def main():
    folder = ROOT / "boundary"
    task = (folder / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same task: {status}")
    torch.set_num_threads(2)
    reports, states, contexts = {}, {}, {}
    for side in ("baseline", "candidate"):
        reports[side] = report = json.loads((folder / side / "report.json").read_text())
        contexts[side] = context = json.loads((folder / side / "run_context.json").read_text())
        if context["source"] != str(ROOT.parents[4] / f".cache/csa-score-single-root-d93bba14-{side}"):
            raise ValueError("Unexpected frozen boundary source")
        if (report["status"], report["batch"], report["history"], report["variant"],
                report["deterministic_level"], report["effective_weight_nz_mode"],
                report["pto_reduction"]["atomic_add"]) != (
                "MEASURED", 4, 65535, "pkg:dsv4_csa_score_single_root_d93bba14", 1, 2, 0):
            raise ValueError(f"Wrong boundary configuration: {side}")
        if any(v["status"] != "PASS" for v in report["pto_self"].values()):
            raise ValueError(f"PTO self-replay differs: {side}")
        if any(v["status"] != "PASS" for checks in report["pto_guards"] for v in checks.values()):
            raise ValueError(f"PTO guard violation: {side}")
        padding = report["padding_graph"]
        if padding["status"] != "PASS" or [r["active_batch"] for r in padding["replays"]] != [4, 3, 1, 4]:
            raise ValueError(f"Padding graph coverage incomplete: {side}")
        if report["topk_selection"]["structural_errors"]:
            raise ValueError(f"Top-K structure failed: {side}")
        states[side] = torch.load(folder / side / "states.pt", map_location="cpu", weights_only=True)["pto"]
        if set(states[side]) != STATES:
            raise ValueError(f"Incomplete state coverage: {side}")
    for key in ("checkpoint", "seed", "layer_index", "accuracy_fixture"):
        if reports["baseline"][key] != reports["candidate"][key]:
            raise ValueError(f"Boundary A/B configuration differs: {key}")
    for key in ("device", "cann", "variant"):
        if contexts["baseline"][key] != contexts["candidate"][key]:
            raise ValueError(f"Boundary runtime differs: {key}")
    checks = {name: compare_tensor(states["candidate"][name], states["baseline"][name], 0, 0)
              for name in sorted(STATES)}
    result = {
        "task": task, "task_status": status,
        "scope": "B4/S6/T24, history65535, active-B 4/3/1/4; odd 1024-step tail and single-root padding; no EP16 claim",
        "checks": checks,
        "contexts": contexts,
        "padding": {s: compact_padding(r["padding_graph"], folder / s / "report.json")
                    for s, r in reports.items()},
        "status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL",
    }
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"])
    if result["status"] != "PASS":
        raise SystemExit("Cross-version boundary state comparison failed")


if __name__ == "__main__":
    main()
