"""把单档设备尾部与主机准备、线程CPU时间和GC放在同一正式步中核对。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_qa_matrix_20260928/arrival.py"
    spec = importlib.util.spec_from_file_location("arrival", path)
    arrival = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arrival)
    arrival.ROOT = ROOT
    arrival.main()
    arrival_cases = json.loads((ROOT / "arrival.json").read_text())["cases"]
    source = json.loads((ROOT / "model/forward.json").read_text())
    result = {"scope": "同一次正式10步；主机monotonic_ns可在本机跨进程比较，thread_cpu_ns只在同一线程内作差。",
              "limits": "主机墙钟减线程CPU时间含阻塞、等待和调度，不能只称CPU抢占；GC重合是观测，不自动证明唯一根因。",
              "cases": []}
    lines = ["# forward入场主机诊断", "", result["scope"], "", result["limits"], "",
             "正式性能见[RESULTS.md](model/RESULTS.md)，设备相对进入异常见[ARRIVAL.md](ARRIVAL.md)。", "",
             "| 档位 | 侧 | Worker冻结对象数范围 | 正式窗口GC次数/最大ms | "
             "最慢准备墙钟/线程CPU ms | 最慢forward提交墙钟ms |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    tail_rows = []
    for case in source["cases"]:
        item = {"history": case["history"], "batch": case["batch"]}
        for side in ("native", "pto"):
            folder = ROOT / "model" / f"h{case['history']}" / f"b{case['batch']}" / side
            ranks = []
            for rank in range(16):
                record = json.loads((folder / f"rank{rank}.performance.json").read_text())
                if not record.get("forward_host_diagnostics"):
                    raise ValueError("该档未开启主机诊断")
                steady = record["steady_window"][0]
                host = steady["host_diagnostics"]
                gc_start, collections = {}, []
                for event in host["gc_events"]:
                    key = (event["generation"], event["thread_id"])
                    if event["phase"] == "start":
                        gc_start[key] = event
                    else:
                        start = gc_start.pop(key)
                        collections.append({"generation": event["generation"], "thread_id": event["thread_id"],
                                            "start_ns": start["monotonic_ns"], "end_ns": event["monotonic_ns"],
                                            "duration_ms": (event["monotonic_ns"]-start["monotonic_ns"])/1e6,
                                            "collected": event["collected"]})
                steps = []
                for index, entry in enumerate(host["steps"]):
                    begin, forward, submitted, end = (entry[k] for k in (
                        "execute_entry", "forward_entry", "forward_submitted", "execute_return"))
                    row = {"step": entry["step"], "forward_us": steady["forward"]["samples_us"][index],
                           "execute_entry_ns": begin["monotonic_ns"], "forward_entry_ns": forward["monotonic_ns"],
                           "pre_wall_ms": (forward["monotonic_ns"]-begin["monotonic_ns"])/1e6,
                           "pre_thread_cpu_ms": (forward["thread_cpu_ns"]-begin["thread_cpu_ns"])/1e6,
                           "submit_wall_ms": (submitted["monotonic_ns"]-forward["monotonic_ns"])/1e6,
                           "submit_thread_cpu_ms": (submitted["thread_cpu_ns"]-forward["thread_cpu_ns"])/1e6,
                           "forward_submitted_ns": submitted["monotonic_ns"],
                           "execute_return_ns": end["monotonic_ns"],
                           "overlapping_gc": [g for g in collections if g["start_ns"] < end["monotonic_ns"]
                                              and g["end_ns"] > begin["monotonic_ns"]]}
                    steps.append(row)
                measured_gc = [g for g in collections if g["start_ns"] < steps[-1]["execute_return_ns"]
                               and g["end_ns"] > steps[0]["execute_entry_ns"]]
                ranks.append({"rank": rank, "gc_freeze_count": host["gc_freeze_count"],
                              "gc_enabled": host["gc_enabled"], "gc_threshold": host["gc_threshold"],
                              "gc_intervals": collections, "measured_window_gc": measured_gc, "steps": steps})
            # All host timestamps use the same machine monotonic clock.
            for index in range(10):
                for phase in ("execute_entry", "forward_entry", "forward_submitted", "execute_return"):
                    center = statistics.median(r["steps"][index][f"{phase}_ns"] for r in ranks)
                    for rank in ranks:
                        row = rank["steps"][index]
                        row[f"{phase}_minus_rank_median_ms"] = (row[f"{phase}_ns"]-center)/1e6
            item[side] = ranks
            device = next(c for c in arrival_cases if (c["history"], c["batch"]) ==
                          (case["history"], case["batch"]))[side]
            for sample in device["samples"]:
                rank = sample["late_rank"]
                delay = sample["relative_entry_excursion_us"][rank] / 1000
                if delay <= 2:
                    continue
                host_step = next(s for s in ranks[rank]["steps"] if s["step"] == sample["step"])
                typical_pre = statistics.median(s["pre_wall_ms"] for s in ranks[rank]["steps"])
                gc_overlap = sum(g["duration_ms"] for g in host_step["overlapping_gc"])
                tail_rows.append(f"| {case['history']//1024}K/B{case['batch']} | {side} | {sample['step']} | {rank} | "
                                 f"{delay:.3f} | {host_step['execute_entry_minus_rank_median_ms']:.3f} | "
                                 f"{host_step['forward_entry_minus_rank_median_ms']:.3f} | "
                                 f"{host_step['pre_wall_ms']:.3f}/{typical_pre:.3f} | {gc_overlap:.3f} |")
            all_steps = [(r["rank"], s) for r in ranks for s in r["steps"]]
            _, pre = max(all_steps, key=lambda value: value[1]["pre_wall_ms"])
            freeze = [r["gc_freeze_count"] for r in ranks]
            measured_gc = [g for r in ranks for g in r["measured_window_gc"]]
            gc_max = max((g["duration_ms"] for g in measured_gc), default=0)
            submit = max(s["submit_wall_ms"] for _, s in all_steps)
            lines.append(f"| {case['history']//1024}K/B{case['batch']} | {side} | {min(freeze)}–{max(freeze)} | "
                         f"{len(measured_gc)}/{gc_max:.3f} | {pre['pre_wall_ms']:.3f}/{pre['pre_thread_cpu_ms']:.3f} | "
                         f"{submit:.3f} |")
        result["cases"].append(item)
    lines += ["", "相对设备入场异常超过2ms的步骤（诊断筛选，不是验收阈值）：", "",
              "| 档位 | 侧 | step | rank | 设备异常ms | 主机execute相对中位ms | "
              "主机forward相对中位ms | 准备/通常ms | 同execute内GC ms |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *tail_rows]
    if not tail_rows:
        lines += ["", "本次未出现超过2ms的设备入场异常；不能由未复现宣称先前长尾已修复。"]
    lines += ["", "逐rank、逐step的原始主机时间、CPU差值、GC区间及设备forward见[host.json](host.json)。",
              "表中GC只统计首个正式execute_entry到最后正式execute_return之间（含步间）的观测，"
              "其他GC仍保留在JSON，不用预热期GC解释正式窗口长尾。",
              "没有修改GC策略，也没有把准备耗时从正式结果中扣除。"]
    (ROOT / "host.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "HOST.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
