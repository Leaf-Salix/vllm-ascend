"""复用入场诊断，并把既有同步、输入、metadata和提交各段对应到迟到rank。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = {
    "input_sync": ("input_sync_begin", "input_sync_end"),
    "state_update_gap": ("input_sync_end", "inputs_begin"),
    "inputs": ("inputs_begin", "inputs_end"),
    "batch_coordination": ("batch_coordination_begin", "batch_coordination_end"),
    "attention_metadata": ("attention_metadata_begin", "attention_metadata_end"),
    "preprocess": ("preprocess_begin", "preprocess_end"),
    "forward_context_gap": ("preprocess_end", "forward_entry"),
    "observer_prepare": ("forward_entry", "event_record_ready"),
    "event_and_submit": ("event_record_ready", "forward_submitted"),
}


def main():
    source = ROOT.parent / "csa_forward_sequence_20260928/analyze.py"
    spec = importlib.util.spec_from_file_location("host_analysis", source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analyzer.ROOT = ROOT
    analyzer.main()
    source = json.loads((ROOT / "arrival.json").read_text())
    report = {"scope": "同一正式10步的可选主机分项，不从设备forward中扣减；不是kernel计时。",
              "limits": "准备段含已有设备同步/忙等，墙钟与线程CPU之差不能单独证明OS抢占。"
                        "gap是两个既有标记之间的范围，不能自动归因为单一函数。", "cases": []}
    lines = ["# EP16入场分项", "", report["scope"], "", report["limits"], "",
             "下面仅展开设备相对入场异常>2ms的rank；所有正式样本仍用于性能统计。", "",
             "| 档位/侧/step/rank | 设备迟到ms | 主机段 | 当步墙钟/线程CPU ms | 同rank十步墙钟中位ms |",
             "| --- | ---: | --- | ---: | ---: |"]
    for case in source["cases"]:
        item = {"history": case["history"], "batch": case["batch"], "sides": {}}
        for side in ("native", "pto"):
            folder = ROOT / "model" / f"h{case['history']}" / f"b{case['batch']}" / side
            ranks = []
            for rank in range(16):
                raw = json.loads((folder / f"rank{rank}.performance.json").read_text())
                window = raw["steady_window"][0]
                if window["timing_event_setup"] != "prewarm_before_generation":
                    raise ValueError("本轮要求两侧预创建计时事件")
                steps = []
                for entry in window["host_diagnostics"]["steps"]:
                    phases = {}
                    for name, (start, finish) in PHASES.items():
                        # Report missing hooks explicitly; never silently use zeros.
                        if start not in entry or finish not in entry:
                            raise ValueError(f"{folder}/rank{rank}/step{entry['step']}: 缺少 {start}/{finish}")
                        a, b = entry[start], entry[finish]
                        phases[name] = {"wall_ms": (b["monotonic_ns"] - a["monotonic_ns"]) / 1e6,
                                        "thread_cpu_ms": (b["thread_cpu_ns"] - a["thread_cpu_ns"]) / 1e6}
                    steps.append({"step": entry["step"], "phases": phases, "raw_marks": entry})
                ranks.append({"rank": rank, "steps": steps})
            tails = []
            for sample in case[side]["samples"]:
                rank = sample["late_rank"]
                delay = sample["relative_entry_excursion_us"][rank] / 1000
                if delay <= 2:
                    continue
                steps = ranks[rank]["steps"]
                current = next(s for s in steps if s["step"] == sample["step"])
                tail = {"rank": rank, "step": sample["step"], "relative_device_entry_ms": delay, "phases": {}}
                for name, phase in current["phases"].items():
                    typical = statistics.median(s["phases"][name]["wall_ms"] for s in steps)
                    tail["phases"][name] = {**phase, "rank_median_wall_ms": typical}
                    lines.append(f"| {case['history']//1024}K/B{case['batch']}/{side}/{sample['step']}/{rank} | "
                                 f"{delay:.3f} | {name} | {phase['wall_ms']:.3f}/{phase['thread_cpu_ms']:.3f} | "
                                 f"{typical:.3f} |")
                tails.append(tail)
            item["sides"][side] = {"ranks": ranks, "entry_tails": tails}
        report["cases"].append(item)
    lines += ["", "全部rank原始标记与分项：[phases.json](phases.json)。"
              "正式统计：[model/RESULTS.md](model/RESULTS.md)。",
              "未复现不能证明已修复；本轮同时验证KV候选，不能把跨轮变化单独归因于事件预热。"]
    (ROOT / "phases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "PHASES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
