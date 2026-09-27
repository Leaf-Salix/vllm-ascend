"""复用已解析rank0 trace，区分单卡CSA、模型内CSA和正式forward。"""

import json
import statistics
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from offline_pd.performance import device_tasks, distribution, layer_intervals, require  # noqa: E402

CASES = ((131072, 16), (8192, 40))
PHASES = ("main_graph_us", "c4_body_sum_us", "other_attention_sum_us",
          "ffn_sum_us", "outside_half_intervals_us")


def single_profile(path, side):
    data = json.loads(Path(path).read_text())
    events = data if isinstance(data, list) else data["traceEvents"]
    pids = {e["pid"] for e in events if e.get("name") == "process_name"
            and e.get("args", {}).get("name") == "Ascend Hardware"}
    rows = [e for e in events if e.get("pid") in pids and e.get("ph") == "X"]
    if side == "native":
        begin = [e for e in rows if e["name"].startswith("HcPre")]
        end = [e for e in rows if e["name"].startswith("HcPost")]
        require(len(begin) == len(end) == 1, "独立单层profile必须只有一个HC_pre/post区间")
        span = Decimal(str(end[0]["ts"])) + Decimal(str(end[0]["dur"])) - Decimal(str(begin[0]["ts"]))
        return {"source": path, "samples": 1, "csa_us": float(span)}
    runtime = [e for e in rows if e["name"].startswith("simpler_aicpu_kernel_exec_")]
    worker = [e for e in rows if e["name"].startswith("aicore_kernel_mode_")]
    require(len(runtime) == len(worker) == 1, "独立单层profile必须只有一个PTO根调用")
    start = min(Decimal(str(e["ts"])) for e in runtime + worker)
    end = max(Decimal(str(e["ts"])) + Decimal(str(e["dur"])) for e in runtime + worker)
    return {"source": path, "samples": 1, "csa_us": float(end - start),
            "worker_us": float(worker[0]["dur"])}


def main():
    single = json.loads((ROOT.parent / "csa_incore_20260927/final_f76b3ad4/cases.json").read_text())
    singles = {(c["history"], c["batch"]): c for c in single["cases"]}
    forward = json.loads((ROOT / "forward.json").read_text())
    forwards = {(c["history"], c["batch"]): c for c in forward["cases"]}
    report = {
        "operator_revision": "f76b3ad4", "rank": 0,
        "scope": "独立Level0 profile，rank0，3步、每步21个C4区间；不是正式无profiler forward样本。",
        "phase_scope": "主图区间从首个HC_pre到最后HC_post；不冒充完整_model_forward设备事件边界。",
        "limits": ("单卡profile每侧只有1次；模型内为63次，输入/历史来源和cache访问环境不同。"
                   "不能据此分摊正式forward差距。"),
        "cases": [],
    }
    for history, batch in CASES:
        key = (history, batch)
        case = {"history": history, "batch": batch, "rank": 0,
                "unprofiled_forward_all_ranks": forwards[key]["forward"]}
        for side in ("native", "pto"):
            folder = ROOT / f"h{history}" / f"b{batch}" / side
            rows = device_tasks(folder, 0)
            layers = layer_intervals(rows, side)
            stats = distribution([i["us"] for i in layers["intervals"]])
            phases = {k: statistics.mean(s[k] for s in layers["profiled_main_steps"]) for k in PHASES}
            case[side] = {"source": str(folder), "csa": stats, "phases": phases, **layers,
                          "single_unprofiled": singles[key]["timing"][side if side == "native" else "pto_full"],
                          "single_profile": single_profile(singles[key]["profiles"][side], side)}
            if side == "pto":
                pairs = []
                for interval in layers["intervals"]:
                    inside = [r for r in rows if interval["start_ns"] <= r["start_ns"] < interval["end_ns"]]
                    runtime = [r for r in inside if r["name"].startswith("simpler_aicpu_kernel_exec_")]
                    worker = [r for r in inside if r["name"].startswith("aicore_kernel_mode_")]
                    require(len(runtime) == len(worker) == 1, "PTO根调用runtime/worker必须唯一")
                    pairs.append({"worker_us": worker[0]["duration_ns"] / 1000,
                                  "outside_worker_us": interval["body_us"] - worker[0]["duration_ns"] / 1000})
                case[side]["root_breakdown"] = {
                    k: distribution([p[k] for p in pairs]) for k in pairs[0]}
        report["cases"].append(case)
    (ROOT / "model_gap_rank0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for c in report["cases"]:
        print(c["history"], c["batch"], {s: c[s]["csa"] for s in ("native", "pto")})


if __name__ == "__main__":
    main()
