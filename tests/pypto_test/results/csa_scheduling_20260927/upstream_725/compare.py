"""Compare Worker timelines; do not infer scheduler causality from overlap."""

import collections
import importlib.util
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
UPSTREAM = RESULTS / "csa_baseline_20260926/upstream_gap/upstream_worker_trace.json"
CURRENT = RESULTS / "csa_split_optimization_20260927/schedule_c7a52af5/h8192_b16"


def canonical(name):
    name = name.split("(")[0].removesuffix("_spmd")
    # Runtime-exclusive constexpr specializations share the same logical task.
    score = re.fullmatch(r"indexer_score_topk_native_pair(?:_\d+)?_(aic|aiv)", name)
    if score:
        return "indexer_score_topk_native_pair_" + score.group(1)
    for hint in ("indexer_head_coefficients", "indexer_topk_query_merge"):
        if re.fullmatch(re.escape(hint) + r"(?:_+\d+)?", name):
            return hint
    for prefix, alias in (("qr_proj_matmul", "qr_proj_matmul"), ("kv_proj_matmul", "kv_proj_matmul"),
                          ("_proj_b_mm_nz_kernel", "proj_b_mm"), ("proj_a_mm", "proj_a_mm"),
                          ("proj_b_mm", "proj_b_mm"), ("proj_b_act", "proj_b_act"), ("quant", "quant")):
        if name == prefix or re.fullmatch(re.escape(prefix) + r"_+\d+", name):
            return alias
    return name


def summarize(path, publication_task="merge_norm"):
    data = json.loads(path.read_text())
    events = data["traceEvents"]
    pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
            and e.get("args", {}).get("name") == "Worker View"}
    workers = [e for e in events if e.get("pid") in pids and e.get("ph") == "X"
               and "kernel-duration-us" in e.get("args", {})]
    origin = min(e["ts"] for e in workers)
    end = max(e["ts"] + e["dur"] for e in workers) - origin
    groups = collections.defaultdict(list)
    for e in workers:
        groups[canonical(e["name"])].append(e)
    tasks = {}
    for name, rows in groups.items():
        starts = [e["ts"] + e["args"]["local_setup_us"] - origin for e in rows]
        ends = [e["ts"] + e["dur"] - origin for e in rows]
        receives = [e["ts"] - origin for e in rows]
        tasks[name] = {
            "blocks": len(rows), "first_receive_us": min(receives),
            "first_start_us": min(starts), "last_start_us": max(starts), "last_end_us": max(ends),
            "start_spread_us": max(starts) - min(starts), "worker_envelope_us": max(ends) - min(receives),
            "kernel_mean_us": statistics.mean(e["args"]["kernel-duration-us"] for e in rows),
            "setup_mean_us": statistics.mean(e["args"]["local_setup_us"] for e in rows),
        }
    norm_end = tasks["mix_x_rms_norm"]["last_end_us"]
    sparse_receive = min(tasks[k]["first_receive_us"] for k in ("qk_pv_aic", "qk_pv_aiv"))
    publication_end = tasks[publication_task]["last_end_us"]
    assert 0 < norm_end < sparse_receive < publication_end <= end
    publication_key = "merge_end" if publication_task == "merge_norm" else "publication_end"
    phases = {
        "first_worker_to_norm_end": norm_end,
        "norm_end_to_sparse_receive": sparse_receive - norm_end,
        f"sparse_receive_to_{publication_key}": publication_end - sparse_receive,
        f"{publication_key}_to_last_worker": end - publication_end,
    }
    return {"path": str(path), "metadata": data.get("metadata", {}), "worker_span_us": end,
            "worker_instances": len(workers), "phases_us": phases, "tasks": tasks,
            "publication_task": publication_task}


def load_timing():
    path = RESULTS / "csa_native_cube_matrix_20260927/summarize.py"
    spec = importlib.util.spec_from_file_location("timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize_timing(CURRENT / "timing/report.json")


def span(values):
    lo, hi = min(values), max(values)
    return f"{lo:.2f}" if lo == hi else f"{lo:.2f}–{hi:.2f}"


def main():
    report = json.loads((CURRENT / "swimlane/report.json").read_text())
    up = summarize(UPSTREAM)
    current = [summarize(Path(w["merged_swimlane"])) for w in report["swimlane_windows"]]
    out = {
        "scope": "Historical 727.98 us Worker reference versus c7a52af5 8K/B16/S6/TP1, four DFX windows.",
        "limits": ("Historical source/configuration and Scheduler View are missing; not same-input A/B. "
                   "Kernel time includes in-core waits. Worker setup is not pure scheduler overhead."),
        "source_reference": ("Latest pypto-lib upstream/main; record fetched revision in README "
                             "separately from historical trace."),
        "current_source": "c7a52af5", "upstream": up, "current": current, "timing": load_timing(),
    }
    (ROOT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 727.98 μs 上游 Worker 泳道与当前 8K/B16 对照", "",
             "当前为 c7a52af5、4个独立DFX窗口；各自首个Worker receive归零。上游配置不完整，不能视为同输入A/B。",
             f"无profiler本体均值 {out['timing']['body']['mean_us']:.2f} μs，"
             f"p50 {out['timing']['body']['p50_us']:.2f} μs；与DFX首尾窗口分开。", "",
             "## 不重叠分段", "", "分段可以相加为单个窗口的Worker首尾耗时；各列范围端点来自不同窗口，不能再相加。", "",
             "| 区间 μs | 历史上游 | 当前4窗口 |", "| --- | ---: | ---: |"]
    labels = ("首Worker→norm结束", "norm结束→Sparse首receive", "Sparse首receive→merge结束", "merge结束→末Worker")
    for label, key in zip(labels, up["phases_us"]):
        lines.append(f"| {label} | {up['phases_us'][key]:.2f} | {span([c['phases_us'][key] for c in current])} |")
    lines += [f"| 总Worker窗口 | {up['worker_span_us']:.2f} | {span([c['worker_span_us'] for c in current])} |", "",
              "## 逐任务对照", "", "单元格均为上游 → 当前范围，—表示该实现无同名任务。任务间窗口有重叠，不可求和。", "",
              "| Task | Worker数量 | 核内均值 μs | kernel启动分散 μs | 平均setup μs | "
              "首次kernel开始 μs | 最晚kernel结束 μs |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    keys = dict.fromkeys([*up["tasks"], *(k for c in current for k in c["tasks"])])
    fields = ("blocks", "kernel_mean_us", "start_spread_us", "setup_mean_us", "first_start_us", "last_end_us")
    for name in keys:
        a = up["tasks"].get(name)
        bs = [c["tasks"][name] for c in current if name in c["tasks"]]
        cells = []
        for field in fields:
            old = (str(a[field]) if field == "blocks" else f"{a[field]:.2f}") if a else "—"
            new = span([b[field] for b in bs]) if bs else "—"
            cells.append(old + " → " + new)
        lines.append("| " + " | ".join([name, *cells]) + " |")
    lines += ["", "## 原始文件", "", f"- 历史上游：`{UPSTREAM}`"]
    lines.extend(f"- 当前窗口{i}：`{c['path']}`" for i, c in enumerate(current))
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"upstream": up["phases_us"], "current": [c["phases_us"] for c in current],
                      "timing": out["timing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
