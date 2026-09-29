"""Validate the affected T18/short-history/padding boundary, without device work."""

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
        if context["source"] != str(ROOT.parents[4] / f".cache/csa-sparse-final-publish-fix-4ffccb7b-{side}"):
            raise ValueError("Unexpected frozen boundary source")
        if (report["status"], report["batch"], report["history"], report["variant"],
                report["deterministic_level"], report["effective_weight_nz_mode"],
                report["pto_reduction"]["atomic_add"]) != (
                "MEASURED", 3, 127, "pkg:dsv4_csa_sparse_final_publish_4ffccb7b", 1, 2, 0):
            raise ValueError(f"Wrong boundary configuration: {side}")
        if any(v["status"] != "PASS" for v in report["pto_self"].values()):
            raise ValueError(f"PTO self-replay differs: {side}")
        if any(v["status"] != "PASS" for checks in report["pto_guards"] for v in checks.values()):
            raise ValueError(f"PTO guard violation: {side}")
        padding = report["padding_graph"]
        if padding["status"] != "PASS" or [r["active_batch"] for r in padding["replays"]] != [3, 2, 1, 3]:
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
        "scope": "B3/S6/T18, history127, active-B 3/2/1/3; no timing or EP16 claim",
        "checks": checks,
        "contexts": contexts,
        "padding": {s: r["padding_graph"] for s, r in reports.items()},
        "status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL",
    }
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"])
    if result["status"] != "PASS":
        raise SystemExit("Cross-version boundary state comparison failed")


if __name__ == "__main__":
    main()
