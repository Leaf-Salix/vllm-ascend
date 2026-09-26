# SPDX-License-Identifier: Apache-2.0
"""离线核对 performance 命令的完整步设备样本、token 与 DSpark。"""

import argparse
import gzip
import json
import math
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from offline_pd.compare import _spec_stats


def distribution(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("设备耗时缺失或含非正/非有限数")
    ordered = sorted(values)
    return {"samples": len(values), "min_us": ordered[0], "p50_us": statistics.median(values),
            "p95_us": ordered[math.ceil(len(values) * 0.95) - 1], "max_us": ordered[-1]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def device_tasks(directory, rank):
    """保留设备原始时间、图/流/任务编号，去掉重复 CPU 栈与可再生分析产物。"""
    target = directory / "device_tasks" / f"rank{rank}.json.gz"
    if target.exists():
        with gzip.open(target, "rt") as handle:
            return json.load(handle)
    paths = list((directory / "trace" / f"rank{rank}").glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    require(len(paths) == 1, f"{directory}/rank{rank}: 缺少唯一已解析设备 trace")
    data = json.loads(paths[0].read_text())
    events = data if isinstance(data, list) else data["traceEvents"]
    device_pids = {item["pid"] for item in events if item.get("name") == "process_name" and
                   item.get("args", {}).get("name") == "Ascend Hardware"}
    require(len(device_pids) == 1, f"rank{rank}: TP1 trace 必须只有一个硬件进程")
    rows = []
    for event in events:
        if event.get("pid") not in device_pids or event.get("ph") != "X":
            continue
        args = event.get("args", {})
        rows.append({"name": event["name"], "start_ns": int(Decimal(str(event["ts"])) * 1000),
                     "duration_ns": int(Decimal(str(event["dur"])) * 1000),
                     "model": args.get("Model Id"), "stream": event["tid"],
                     "task": args.get("Task Id"), "batch": args.get("Batch Id"),
                     "subtask": args.get("Subtask Id"), "kind": args.get("Task Type")})
    require(bool(rows), f"rank{rank}: 设备 trace 为空")
    rows.sort(key=lambda item: item["start_ns"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"))
    return rows


def layer_intervals(rows, side, steps=3):
    """按主图及固定模型的 attention/FFN 次序映射，计数或次序不符即拒绝。"""
    targets = set(range(2, 43, 2))
    native_per_step = 86 - (len(targets) if side == "pto" else 0)
    counts = Counter(row["model"] for row in rows if row["name"] == "HcPre")
    models = [model for model, count in counts.items()
              if model not in (None, 4294967295) and count == steps * native_per_step]
    require(len(models) == 1, f"{side}: 不能唯一识别 43 层主图，HcPre 图计数={dict(counts)}")
    model = models[0]
    pre = [row for row in rows if row["model"] == model and row["name"] == "HcPre"]
    post = [row for row in rows if row["model"] == model and row["name"] == "HcPost"]
    require(len(pre) == len(post), f"{side}: HC_pre/post 不配对")

    def end(row):
        return row["start_ns"] + row["duration_ns"]

    halves = []
    for begin, finish in zip(pre, post):
        require(begin["start_ns"] < finish["start_ns"], f"{side}: HC_pre/post 次序错误")
        halves.append({"kind": "native", "start_ns": begin["start_ns"], "end_ns": end(finish),
                       "first_task": begin["task"], "last_task": finish["task"]})
    intervals, model_steps = [], []
    expected_order = [kind for layer in range(43)
                      for kind in ("pto" if side == "pto" and layer in targets else "native", "native")]
    for step in range(steps):
        segment = halves[step * native_per_step:(step + 1) * native_per_step]
        lower, upper = segment[0]["start_ns"], segment[-1]["end_ns"]
        in_step = [row for row in rows if lower <= row["start_ns"] < upper]
        if side == "pto":
            runtimes = [row for row in in_step if row["name"].startswith("simpler_aicpu_kernel_exec_")]
            workers = [row for row in in_step if row["name"].startswith("aicore_kernel_mode_")]
            require(len(runtimes) == len(workers) == 21,
                    f"step{step}: PTO runtime/worker 数量不符：{len(runtimes)}/{len(workers)}")
            for runtime, worker in zip(runtimes, workers):
                require(max(runtime["start_ns"], worker["start_ns"]) < min(end(runtime), end(worker)),
                        f"step{step}: runtime/worker 未重叠，不能配对")
                segment.append({"kind": "pto", "start_ns": min(runtime["start_ns"], worker["start_ns"]),
                                "end_ns": max(end(runtime), end(worker)), "first_task": runtime["task"],
                                "last_task": worker["task"]})
        segment.sort(key=lambda item: item["start_ns"])
        require([item["kind"] for item in segment] == expected_order,
                f"step{step}: attention/FFN 次序与正式 43 层模型不符")
        phase_us = [(item["end_ns"] - item["start_ns"]) / 1000 for item in segment]
        model_steps.append({"step": step, "main_graph_us": (upper - lower) / 1000,
                            "c4_body_sum_us": sum(phase_us[2 * layer] for layer in targets),
                            "other_attention_sum_us": sum(phase_us[2 * layer] for layer in range(43)
                                                          if layer not in targets),
                            "ffn_sum_us": sum(phase_us[1::2]),
                            "outside_half_intervals_us": (upper - lower) / 1000 - sum(phase_us)})
        for layer in sorted(targets):
            index = 2 * layer
            item = segment[index]
            start = item["start_ns"]
            # 首次 compact metadata 可能在 PTO 根 kernel 前生产；复用层不得重复计费。
            leading = [row for row in in_step if row["name"] == "CompressorMetadata" and
                       segment[index - 1]["end_ns"] <= row["start_ns"] and end(row) <= start]
            if side == "pto" and leading:
                start = min(row["start_ns"] for row in leading)
            intervals.append({"step": step, "layer": layer, "start_ns": start, "end_ns": item["end_ns"],
                              "us": (item["end_ns"] - start) / 1000,
                              "body_us": (item["end_ns"] - item["start_ns"]) / 1000,
                              "leading_metadata_tasks": len(leading) if side == "pto" else 0,
                              "first_task": item["first_task"], "last_task": item["last_task"]})
    return {"graph_model_id": model, "hc_pre_by_model": dict(counts), "intervals": intervals,
            "profiled_main_steps": model_steps}


def load_rank(root, side, rank, mode, plan, *, batch, tokens, steps):
    path = root / side / f"rank{rank}.performance.json"
    value = json.loads(path.read_text())
    expected_tokens = batch * (plan["decode"]["speculative_tokens"] + 1)
    expected = {"command": "performance", "backend": side, "rank": rank, "batch": batch,
                "expected_tokens": expected_tokens, "weight_nz_mode": mode, "decode_tokens": tokens,
                "graph_mode": "full_decode_only", "stage": "measured", "variant": "performance"}
    for key, item in expected.items():
        require(value.get(key) == item, f"{path}: {key} 与声明不符")
    keys = [c["key"] for c in plan["cases"] if c["p_dp_rank"] == rank % plan["prefill"]["dp"]]
    require(len(keys) == 1 and value["key"] == keys[0], f"{path}: 必须是 bank 声明的唯一 case")
    for name in ("steady_output_token_ids", "output_token_ids"):
        rows = value.get(name)
        require(isinstance(rows, list) and len(rows) == batch and all(
            isinstance(row, list) and len(row) == tokens and all(type(t) is int and t >= 0 for t in row)
            for row in rows), f"{path}: {name} 缺少请求/token")
    require(len(value["steady_window"]) == 1, f"{path}: TP1 只应返回一个 worker")
    steady = value["steady_window"][0]
    require(value.get("eplb_enabled") is False and value.get("dynamic_eplb_env") == "false" and
            value.get("expert_map_record_env") == "false", f"{path}: 所有 CSA 测试必须关闭 EPLB")
    require(steady["dp_rank"] == rank and steady["sufficient"], f"{path}: 稳态设备采样不足")
    samples = steady["device"]["samples_us"]
    stamps = steady["device"]["start_timestamps_raw"]
    count = steady["measured_steps"]
    require(len(samples) == len(stamps) == count and count >= 20 and all(
        b > a for a, b in zip(stamps, stamps[1:])), f"{path}: 设备事件未更新或数量不符")
    require(steady["step_tokens"] == [expected_tokens] * count and
            steady["step_requests"] == [batch] * count, f"{path}: 设备样本混入其他档位")
    distribution(samples)
    require(len(value["window"]) == 1, f"{path}: 缺少唯一 trace 窗口")
    window = value["window"][0]
    require(window["sufficient"] and window["profiled_steps"] == steps and len(window["window"]) == steps,
            f"{path}: trace 样本不足")
    require(all(s["scheduled_tokens"] == expected_tokens and s["requests"] == batch
                for s in window["window"]), f"{path}: trace 档位不符")
    stats = _spec_stats(value["spec_decode"], plan["decode"]["speculative_tokens"])
    if side == "pto":
        observed = value.get("csa_observation", [])
        require(len(observed) == 1, f"{path}: 缺少捕获路径")
        captured = observed[0]["capture_time_selection"]
        # 固定正式模型 43 层中的 21 个 C4 目标层，不能从已有键反推覆盖范围。
        require(set(captured) == {f"model.layers.{i}.self_attn.attn" for i in range(2, 43, 2)},
                f"{path}: C4 捕获层键与正式模型前缀不符")
        require(all(counts.get(f"pto_tokens{expected_tokens}", 0) > 0 for counts in captured.values()),
                f"{path}: 目标档位未实际捕获 PTO")
    return value, stats


def compare(root, mode, plan, *, batch=16, tokens=192, steps=3, ranks=16):
    report = {"status": "FAIL", "mode": mode, "scope": "无 profiler 的 execute_model 设备区间不覆盖"
              "随后 sample_tokens 中的草稿；完整 decode 周期单列。CSA 层区间独立取设备 trace 首末。",
              "expected": {"ranks": ranks, "batch": batch, "tokens_per_round": tokens, "profile_steps": steps},
              "errors": [], "token_mismatches": 0, "spec_decode_mismatched_ranks": 0,
              "compared_tokens": 0, "ranks": []}
    samples = {side: [] for side in ("native", "pto")}
    cycles = {side: [] for side in ("native", "pto")}
    layer_samples = {side: [] for side in ("native", "pto")}
    for rank in range(ranks):
        try:
            loaded = {side: load_rank(root, side, rank, mode, plan, batch=batch, tokens=tokens, steps=steps)
                      for side in ("native", "pto")}
            native, pto = (loaded[side][0] for side in ("native", "pto"))
            for key in ("key", "history", "capture_sizes", "deterministic", "hccl_deterministic", "atomic_add",
                        "eplb_enabled", "dynamic_eplb_env", "expert_map_record_env", "custom_opp_path"):
                require(native[key] == pto[key], f"rank{rank}: 两侧 {key} 不同")
            require(native["window"][0]["window"] == pto["window"][0]["window"],
                    f"rank{rank}: 两侧 trace 步序或形状不同")
            mismatches = sum(a != b for name in ("steady_output_token_ids", "output_token_ids")
                             for left, right in zip(native[name], pto[name]) for a, b in zip(left, right))
            stats_equal = loaded["native"][1] == loaded["pto"][1]
            report["token_mismatches"] += mismatches
            report["spec_decode_mismatched_ranks"] += not stats_equal
            report["compared_tokens"] += 2 * batch * tokens
            entry = {"rank": rank, "key": native["key"], "token_mismatches": mismatches,
                     "spec_decode_equal": stats_equal}
            for side, (data, stats) in loaded.items():
                steady = data["steady_window"][0]
                device = steady["device"]["samples_us"]
                samples[side].extend(device)
                entry[side] = {"execute_model_device": distribution(device), "spec_decode": stats,
                               "peak_allocated_bytes": steady["peak_allocated_bytes"],
                               "peak_reserved_bytes": steady["peak_reserved_bytes"]}
                cycle = steady.get("decode_cycle", {})
                if cycle.get("sufficient"):
                    entry[side]["decode_cycle_device"] = distribution(cycle["samples_us"])
                    cycles[side].extend(cycle["samples_us"])
                layers = layer_intervals(device_tasks(root / side, rank), side, steps)
                entry[side]["layers"] = layers
                layer_samples[side].extend(layers["intervals"])
            report["ranks"].append(entry)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append(f"rank{rank}: {exc}")
    if len(report["ranks"]) == ranks:
        report["execute_model_device"] = {side: distribution(values) for side, values in samples.items()}
        complete_cycles = all("decode_cycle_device" in item[side]
                              for item in report["ranks"] for side in cycles)
        report["decode_cycle_device"] = ({side: distribution(values) for side, values in cycles.items()}
                                          if complete_cycles else None)
        report["decode_cycle_note"] = (
            "完整周期来自相邻起点的 elapsed_time，包含随后采样和草稿。" if complete_cycles else
            "本次采集源码 334c4252 只记录 execute_model 首尾；没有保存完整周期，"
            "不能从原始计数猜测时钟单位，也不能据此判定整模型吞吐/整步加速。")
        report["csa"] = {}
        for side, values in layer_samples.items():
            report["csa"][side] = {
                "all": distribution([v["us"] for v in values]),
                "first_layer": distribution([v["us"] for v in values if v["layer"] == 2]),
                "following_layers": distribution([v["us"] for v in values if v["layer"] != 2]),
                "by_layer": {str(layer): distribution([v["us"] for v in values if v["layer"] == layer])
                             for layer in range(2, 43, 2)},
            }
        report["csa_scope"] = ("独立 Level0 trace 的设备首末区间，包含内部间隙；PTO runtime/worker "
                               "取并集首末，不求和，首次根调用前的 compact metadata 单独列明并纳入。")
        report["csa_pto_p50_below_750us"] = report["csa"]["pto"]["all"]["p50_us"] < 750
    if not report["errors"] and not report["token_mismatches"] and not report["spec_decode_mismatched_ranks"]:
        report["status"] = "MEASURED_TOKEN_PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--mode", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    report = compare(args.root, args.mode, json.loads((args.bank / "plan.json").read_text()))
    (args.root / "performance_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "ranks"}, ensure_ascii=False, indent=2))
    raise SystemExit(report["status"] == "FAIL")


if __name__ == "__main__":
    main()
