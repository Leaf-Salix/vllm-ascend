"""将同轮forward事件与完整主流图跨度配对，保留不同轮正式成绩的界限。"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from offline_pd.performance import device_tasks, layer_intervals, require  # noqa: E402


def full_graph_spans(rows, model, steps):
    graph_rows = [row for row in rows if row["model"] == model]
    first = graph_rows[0]
    stream_rows = [row for row in graph_rows if row["stream"] == first["stream"]]
    starts = [i for i, row in enumerate(stream_rows)
              if (row["name"], row["task"]) == (first["name"], first["task"])]
    require(len(starts) == steps, "完整主图的开始标记与profile窗口数不同")
    parts = [stream_rows[start:(starts[i + 1] if i + 1 < steps else len(stream_rows))]
             for i, start in enumerate(starts)]
    keys = [[(row["name"], row["task"]) for row in part] for part in parts]
    require(all(key == keys[0] for key in keys), "主流任务序列不一致，不能按序配对forward")
    require(all(part[-1]["name"] == "NOTIFY_RECORD" for part in parts), "完整主流图末标记缺失")
    return [{"stream": first["stream"], "tasks": len(part), "first": part[0]["name"],
             "last": part[-1]["name"], "start_ns": part[0]["start_ns"],
             "end_ns": max(row["start_ns"] + row["duration_ns"] for row in part),
             "span_us": (max(row["start_ns"] + row["duration_ns"] for row in part)
                         - part[0]["start_ns"]) / 1000} for part in parts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "model")
    parser.add_argument("--trace-only", action="store_true", help="旧profile只核对完整主图和HC边界，不推算事件差额")
    args = parser.parse_args()
    folder = args.root / "h131072/b8"
    result = {"scope": "rank0/128K/B8同一次profile中的forward事件与完整主流图；正式10步来自另一轮，单列。",
              "limits": "图外差额尚未拆成主机发射等待或设备队列等待；raw时间戳不跨时钟换算；不以profile作为正式成绩。",
              "trace_only": args.trace_only, "sides": {}}
    positions = []
    for side in ("native", "pto"):
        source = folder / side
        measured = json.loads((source / "rank0.performance.json").read_text())
        window = measured["window"][0]
        steps = window["profiled_steps"]
        require(window["sufficient"], f"{side}: profile窗口不完整")
        rows = device_tasks(source, 0)
        intervals = layer_intervals(rows, side, steps=steps)
        graph = full_graph_spans(rows, intervals["graph_model_id"], steps)
        data = {"source": str(source), "event_work_mode": measured["worker_runtime_config"][0]["cann_event_work_mode"],
                "full_graph": graph, "hc_boundary_main_us": [row["main_graph_us"]
                                                            for row in intervals["profiled_main_steps"]],
                "formal_other_round_us": measured["steady_window"][0]["forward"]["samples_us"]}
        data["graph_outside_hc_us"] = [row["span_us"] - hc for row, hc in zip(graph, data["hc_boundary_main_us"])]
        if not args.trace_only:
            event = window["profile_forward"]
            require(event["sufficient"] and len(event["samples_us"]) == steps, f"{side}: 同轮事件缺失")
            require(event["step_indices"] == [row["steady_step_index"] for row in window["window"]],
                    f"{side}: 同轮事件与trace步序不符")
            data["same_round_event"] = event
            data["event_minus_full_graph_us"] = [value - row["span_us"]
                                                 for value, row in zip(event["samples_us"], graph)]
            positions.append(event["positions_cpu"])
        result["sides"][side] = data
        print(side, "full graph ms", statistics.mean(row["span_us"] for row in graph) / 1000,
              "graph outside HC us", statistics.mean(data["graph_outside_hc_us"]),
              "same-round event gap", data.get("event_minus_full_graph_us"))
    if positions:
        result["cpu_positions_equal"] = positions[0] == positions[1]
    (folder / "boundary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
