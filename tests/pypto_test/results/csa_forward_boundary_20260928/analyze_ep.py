"""复用既有16卡trace，按MoE结束时间配对全局轮次，不补设备执行。"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from offline_pd.performance import device_tasks, layer_intervals, require  # noqa: E402

END_MATCH_NS = 100_000


def summarize(values):
    values = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1],
            "max": max(values)} if values else None


def records(folder, side, rank):
    rows = device_tasks(folder / side, rank)
    layers = layer_intervals(rows, side)
    model = layers["graph_model_id"]
    pre = [r for r in rows if r["model"] == model and r["name"].split("_")[0] == "HcPre"]
    post = [r for r in rows if r["model"] == model and r["name"].split("_")[0] == "HcPost"]
    dispatch = [r for r in rows if r["model"] == model
                and r["name"].split("_")[0] == "MoeDistributeDispatchV2"]
    require(len(dispatch) == 3 * 43, f"{side}/rank{rank}: dispatch数量不符")
    csa = {(r["step"], r["layer"]): r for r in layers["intervals"]}
    result, cursor = [], 0
    for step in range(3):
        for layer in range(43):
            if side != "pto" or layer not in range(2, 43, 2):
                cursor += 1
            start = pre[cursor]["start_ns"]
            end = post[cursor]["start_ns"] + post[cursor]["duration_ns"]
            cursor += 1
            match = [r for r in dispatch if start <= r["start_ns"]
                     and r["start_ns"] + r["duration_ns"] <= end]
            require(len(match) == 1, f"{side}/rank{rank}/step{step}/layer{layer}: dispatch边界不唯一")
            d = match[0]
            row = {"rank": rank, "step": step, "layer": layer,
                   "dispatch_start_ns": d["start_ns"],
                   "dispatch_end_ns": d["start_ns"] + d["duration_ns"]}
            if (step, layer) in csa:
                row["csa"] = csa[step, layer]
            result.append(row)
    require(cursor == len(pre) == len(post), "HC映射未完全覆盖")
    return result


def align(ranks):
    windows, unmatched, used = [], [], set()
    for anchor in ranks[0]:
        matches = []
        for rank, rows in enumerate(ranks):
            candidates = [r for r in rows if r["layer"] == anchor["layer"]
                          and abs(r["dispatch_end_ns"] - anchor["dispatch_end_ns"]) <= END_MATCH_NS]
            if len(candidates) != 1:
                matches = []
                break
            matches.append(candidates[0])
        if not matches:
            unmatched.append({"step": anchor["step"], "layer": anchor["layer"]})
            continue
        for row in matches:
            key = (row["rank"], row["step"], row["layer"])
            require(key not in used, "同一rank的dispatch重复匹配")
            used.add(key)
        base = min(r["dispatch_start_ns"] for r in matches)
        starts = [(r["dispatch_start_ns"] - base) / 1000 for r in matches]
        ends = [(r["dispatch_end_ns"] - base) / 1000 for r in matches]
        category = "first" if anchor["layer"] == 0 else ("c4_following" if "csa" in anchor else "other")
        window = {"anchor_step": anchor["step"], "layer": anchor["layer"], "category": category,
                  "arrival_spread_us": max(starts) - min(starts),
                  "mean_lead_to_last_rank_us": max(starts) - statistics.mean(starts),
                  "mean_dispatch_us": statistics.mean(b - a for a, b in zip(starts, ends)),
                  "mean_after_last_arrival_us": statistics.mean(ends) - max(starts),
                  "end_spread_us": max(ends) - min(ends), "ranks": matches}
        if category == "c4_following":
            csa_ends = [(r["csa"]["end_ns"] - base) / 1000 for r in matches]
            durations = [r["csa"]["us"] for r in matches]
            window.update(csa_end_spread_us=max(csa_ends) - min(csa_ends),
                          csa_mean_us=statistics.mean(durations), csa_max_us=max(durations),
                          csa_max_minus_mean_us=max(durations) - statistics.mean(durations),
                          mean_csa_to_dispatch_us=statistics.mean(a - b for a, b in zip(starts, csa_ends)),
                          csa_end_dispatch_start_correlation=(statistics.correlation(csa_ends, starts)
                                                             if len(set(csa_ends)) > 1 and len(set(starts)) > 1
                                                             else None))
        windows.append(window)
    categories = {}
    for category in ("first", "c4_following", "other"):
        group = [r for r in windows if r["category"] == category]
        keys = [key for key in group[0] if key.endswith("_us")] if group else []
        categories[category] = {key: summarize([r[key] for r in group]) for key in keys}
    return {"matched_windows": len(windows), "unmatched_anchors": unmatched,
            "categories": categories, "windows": windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "model/h131072/b8")
    args = parser.parse_args()
    result = {"scope": "既有独立profile三步、16rank；不同event模式会改变profile扰动，不折算正式forward收益。",
              "method": "按同层dispatch结束与rank0相差不超过100us唯一配对；不假定各rank本地step一致。",
              "sides": {}}
    for side in ("native", "pto"):
        value = align([records(args.folder, side, rank) for rank in range(16)])
        result["sides"][side] = value
        print(side, value["matched_windows"], "matched,", len(value["unmatched_anchors"]), "unmatched")
        print(json.dumps(value["categories"]["c4_following"], ensure_ascii=False))
    (args.folder / "ep_alignment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
