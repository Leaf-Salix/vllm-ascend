"""拆分已采集的主图/FFN任务；不新增设备执行，不代替正式forward计时。"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from offline_pd.performance import device_tasks, distribution, layer_intervals, require  # noqa: E402

PHASES = ("main_graph_us", "c4_body_sum_us", "other_attention_sum_us",
          "ffn_sum_us", "outside_half_intervals_us")


def ffn_tasks(rows, side, model):
    pre = [r for r in rows if r["model"] == model and r["name"].split("_")[0] == "HcPre"]
    post = [r for r in rows if r["model"] == model and r["name"].split("_")[0] == "HcPost"]
    require(len(pre) == len(post), "HC边界不成对")
    cursor = 0
    layers = []
    total, counts = Counter(), Counter()
    for step in range(3):
        for layer in range(43):
            if side != "pto" or layer not in range(2, 43, 2):
                cursor += 1  # Native attention half precedes this FFN.
            start = pre[cursor]["start_ns"]
            end = post[cursor]["start_ns"] + post[cursor]["duration_ns"]
            cursor += 1
            busy = Counter()
            for row in rows:
                if row["start_ns"] >= start and row["start_ns"] + row["duration_ns"] <= end:
                    name = row["name"].split("_")[0]
                    busy[name] += row["duration_ns"] / 1000
                    counts[name] += 1
            total.update(busy)
            layers.append({"step": step, "layer": layer, "span_us": (end - start) / 1000,
                           "busy_us": dict(busy)})
    require(cursor == len(pre), "未完整覆盖43层×3步")
    return {"busy_us_per_step": {k: v / 3 for k, v in total.items()},
            "counts": dict(counts), "layers": layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "model")
    args = parser.parse_args()
    result = {
        "scope": "rank0独立3步Level0，不分摊正式10步forward；两轮重新分配请求cache。",
        "limits": "Native/PTO实际event模式0/1；FFN task busy可能重叠，不是critical span。首层通信包含跨rank到达等待。",
        "cases": [],
    }
    for history, batch in ((131072, 16), (8192, 40)):
        case = {"history": history, "batch": batch}
        breakdown = {"scope": result["scope"] + result["limits"]}
        for side in ("native", "pto"):
            folder = args.root / f"h{history}/b{batch}/{side}"
            rows = device_tasks(folder, 0)
            layers = layer_intervals(rows, side)
            ffn = ffn_tasks(rows, side, layers["graph_model_id"])
            for step in range(3):
                span = sum(r["span_us"] for r in ffn["layers"] if r["step"] == step)
                require(abs(span - layers["profiled_main_steps"][step]["ffn_sum_us"]) < 0.01,
                        f"{history}/{side}/step{step}: FFN分解与半层边界不一致")
            breakdown[side] = ffn
            case[side] = {
                "source": str(folder),
                "csa": distribution([x["us"] for x in layers["intervals"]]),
                "phases": {key: statistics.mean(x[key] for x in layers["profiled_main_steps"])
                           for key in PHASES},
                "first_ffn_mean_us": statistics.mean(x["span_us"] for x in ffn["layers"] if x["layer"] == 0),
                **layers,
            }
            print(history, side, case[side]["phases"], "first FFN", case[side]["first_ffn_mean_us"])
        (args.root / f"h{history}/ffn_breakdown_rank0.json").write_text(
            json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n")
        result["cases"].append(case)
    (args.root / "model_gap_rank0.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
