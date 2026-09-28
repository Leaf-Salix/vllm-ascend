"""Analyze finalized DFX windows without executing or modifying an NPU kernel."""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
HELPER = WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py"


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def annotate_selected_report(path, window):
    missing = {segment["task"] for segment in window["segments"] if segment["untimed_direct_producers"]}
    lines = []
    for line in path.read_text().splitlines():
        line = line.replace(
            "(minimum summed dispatch→finish elapsed)",
            "(preselected window_3; no fastest-window selection)")
        cells = line.split("|")
        if len(cells) == 8 and any(cells[1].strip().startswith(f"`{task}` ") for task in missing):
            cells[2] = cells[3] = " — "
            cells[6] = " 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 "
            line = "|".join(cells)
        lines.append(line)
    note = (
        "> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。\n"
        "> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。\n\n"
    )
    path.write_text(note + "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    case = ROOT / f"h{args.history}_b{args.batch}"
    report = read(case / "swimlane/report.json")
    source = read(ROOT / "source.json")
    if report["status"] != "MEASURED" or report["variant"] != source["variant"]:
        raise ValueError("DFX process has not published the final expected package result")
    if any(read(case / side / "report.json")["status"] != "MEASURED" for side in ("native", "pto")):
        raise ValueError("Wait until both compiled measurements are finalized")
    if len(report["swimlane_windows"]) != 4:
        raise ValueError("Expected four independent windows")
    helper = load("csa_critical_path_helper", HELPER)
    worker = load("csa_worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    out = case / "schedule"
    out.mkdir(exist_ok=True)
    windows = []
    for index, window in enumerate(report["swimlane_windows"]):
        if not window["exported"] or window["execution"] != "graph_replay":
            raise ValueError("Invalid DFX capture")
        trace = Path(window["merged_swimlane"])
        # Uses the official clock-domain join and validates level 4, raw/joined
        # row counts, and every task's logical block count before attribution.
        analysis = helper._build_analysis(trace.parent, trace.parent, 2)
        frequency = analysis.graph.freq
        scale = 1e6 / frequency
        starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
        ends = helper._aggregate_max(analysis.rows_by_task, "end_time_us")
        dispatches = helper._aggregate_min(analysis.rows_by_task, "dispatch_time_us")
        finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
        segments = []
        for segment in analysis.result.segments:
            task = segment.task
            direct = analysis.preds.get(task, set())
            untimed = sorted(pred for pred in direct if not helper._is_alloc(pred, analysis) and pred not in ends)
            timed = [pred for pred in direct if pred in ends]
            # An untimed dummy invalidates complete dependency-ready attribution.
            # Preserve missing producer IDs rather than report partial FIN as readiness.
            data_ready = max((ends[pred] for pred in timed), default=dispatches[task]) if not untimed else None
            observed_ready = max((finishes[pred] for pred in timed), default=dispatches[task]) if not untimed else None
            early, count, total = helper._observed_early(task, analysis, 2 * scale)
            segments.append({
                "task": task, "name": segment.name, "wall_us": segment.dur * scale,
                "observed_contribution_us": segment.compute * scale, "gap_us": segment.stall * scale,
                "gap_kind": segment.kind, "early": early, "early_blocks": count, "blocks": total,
                "dispatch_us": dispatches[task], "start_us": starts[task], "end_us": ends[task],
                "finish_us": finishes[task], "data_ready_us": data_ready,
                "observed_ready_us": observed_ready, "untimed_direct_producers": untimed,
                "producer_end_to_fin_us": observed_ready - data_ready if not untimed else None,
                "fin_to_dispatch_us": dispatches[task] - observed_ready if not untimed else None,
                "dispatch_to_start_us": starts[task] - dispatches[task],
            })
        summary = worker.summarize(trace)
        windows.append({"index": index, "path": str(trace), "joined_rows": len(analysis.rows),
                        "dispatch_to_finish_us": analysis.elapsed_us,
                        "aicore_span_us": analysis.result.makespan * scale,
                        "static_cpm_us": analysis.result.cpm_len * scale,
                        "observed_compute_us": analysis.result.compute_total * scale,
                        "observed_gap_us": analysis.result.stall_total * scale,
                        "segments": segments, "worker": summary})
    # Preselect the final numbered window, independently of measured speed.
    # Passing only that directory prevents the skill's multi-rank fastest selection.
    selected = Path(windows[3]["path"]).parent
    with (out / "critical_path.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "simpler_setup.tools.critical_path", str(selected), "--stdout"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(HELPER), str(selected), "--operator", "decode_csa_tp1_layer",
                        "-o", str(out / "critical_path_summary.md")], check=True, stdout=log, stderr=subprocess.STDOUT)
    annotate_selected_report(out / "critical_path_summary.md", windows[3])
    upstream = worker.summarize(ROOT.parent / "csa_baseline_20260926/upstream_gap/upstream_worker_trace.json")
    result = {"history": args.history, "batch": args.batch, "source": source, "windows": windows,
              "selection": "Window 3 fixed in advance; four replays on one device, not four ranks",
              "upstream": upstream, "scope": "Separate DFX captures; no pure scheduler-cost or Native speed claim",
              "limits": "Untimed dummy producers make readiness incomplete; official helper diagnostics for those "
                        "rows must not be interpreted as complete FIN-to-dispatch attribution"}
    (out / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# {args.history//1024}K/B{args.batch}：四窗口调度诊断", "",
             "当前源码c93ec723；复用已完成的四个level-4窗口，没有追加设备执行。",
             "每窗原始AICore/AICPU/解析行数及每任务block数量均通过官方解析器检查。", "",
             "| 窗口 | 行数 | dispatch→FIN μs | AICore首尾 μs | Static CPM μs | Observed gap μs |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for w in windows:
        lines.append(f"| {w['index']} | {w['joined_rows']} | {w['dispatch_to_finish_us']:.3f} "
                     f"| {w['aicore_span_us']:.3f} | {w['static_cpm_us']:.3f} | {w['observed_gap_us']:.3f} |")
    lines += ["", "完整Observed路径预先固定window_3，不选择最快窗口；[路径和阻塞证据](critical_path_summary.md)。",
              "工具中的compute是逻辑任务对关键路径的覆盖，含SPMD启动跨度和核内等待，不等同纯算术。",
              "Score/qk_pv等MIX任务在路径里以首个AIC函数名显示，但跨度含同一逻辑任务的AIC/AIV；分核计时看Worker表。",
              "dummy缺失物理时戳的任务，完整data-ready/FIN归因置空；不采用工具对这些行的局部ready数值。", "",
              "## Worker非重叠分段", "",
              "历史727.98μs图输入/工具链不完整，且与当前形状不同；仅作结构参照，不是同输入性能对照。", "",
              "| 分段 | 历史上游 μs | 当前四窗口均值 μs |", "| --- | ---: | ---: |"]
    for name, value in upstream["phases_us"].items():
        current = statistics.mean(w["worker"]["phases_us"][name] for w in windows)
        lines.append(f"| {name} | {value:.3f} | {current:.3f} |")
    lines += ["", "所有时间与正式无profiler事件分开；不把两种采样相减，也不把多任务重叠时长相加。",
              "全部任务、实际early派发和缺失前置ID见[evidence.json](evidence.json)。"]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
