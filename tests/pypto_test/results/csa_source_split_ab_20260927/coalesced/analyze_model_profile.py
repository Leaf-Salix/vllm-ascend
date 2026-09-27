"""Read existing rank0 Level0 traces; these are separate from formal forward samples."""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "model"
sys.path.insert(0, str(ROOT.parents[3]))
from offline_pd.performance import device_tasks, distribution, layer_intervals  # noqa: E402


def main():
    report = {"operator_revision": "2a740c1f + coalesced/candidate.patch",
              "scope": "rank0独立3步Level0，不能分摊无profiler正式forward；两轮分别恢复请求cache", "cases": []}
    phases = ("main_graph_us", "c4_body_sum_us", "other_attention_sum_us",
              "ffn_sum_us", "outside_half_intervals_us")
    for history, batch in ((131072, 16), (8192, 40)):
        case = {"history": history, "batch": batch}
        for side in ("native", "pto"):
            folder = ROOT / f"h{history}/b{batch}" / side
            rows = device_tasks(folder, 0)
            layers = layer_intervals(rows, side)
            case[side] = {
                "source": str(folder),
                "csa": distribution([x["us"] for x in layers["intervals"]]),
                "phases": {key: statistics.mean(x[key] for x in layers["profiled_main_steps"]) for key in phases},
                **layers,
            }
            print(history, batch, side, case[side]["csa"], case[side]["phases"])
        report["cases"].append(case)
    (ROOT / "model_gap_rank0.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
