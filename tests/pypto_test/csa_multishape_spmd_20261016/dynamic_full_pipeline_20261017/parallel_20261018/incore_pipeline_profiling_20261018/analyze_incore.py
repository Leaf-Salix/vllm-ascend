#!/usr/bin/env python3
"""核验72核覆盖并整理逐核流水，不用指令周期之和冒充墙钟耗时。

只处理已完成的原版skill导出。所有timestamp原样保留；概览是同一核/同一
pipe内实际计算指令区间的并集，不填补空隙、不把72条核时间各自归零。
原始Scalar/同步事件完整保留在raw_simulator；精度验证是独立的硬门槛。
"""

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from prepare_standalone import HERE


WORKSPACE = HERE.parents[6]
DESTINATION = WORKSPACE / "downloads/CSA_dynamic_128K_B16_S6_all72_incore_model_bounded_20261019"
SYNC = frozenset({"BAR", "SET_FLAG", "WAIT_FLAG", "PIPE_BARRIER", "SYNC_ALL", "WAIT_EVENT"})
COMPUTE_PIPES = frozenset({"CUBE", "VECTOR", "MTE1", "MTE2", "MTE3", "FIXP"})


def core_key(name):
    match = re.fullmatch(r"core(\d+)\.(cubecore|veccore)(\d+)", name)
    assert match, name
    return int(match[1]), 0 if match[2] == "cubecore" else 1 + int(match[3])


def intervals_union(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals)
    result = [list(ordered[0])]
    for begin, end in ordered[1:]:
        if begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def main():
    case = Path((HERE / "ACTIVE_CASE_PATH.txt").read_text().strip())
    bindings = json.loads((case / "workload_bindings.json").read_text())
    assert bindings.get("bounded_model_backoff") is True
    run = case.parents[2]
    manifest = list(csv.DictReader((run / "manifest_export.csv").open()))
    assert len(manifest) == 1 and manifest[0]["status"] == "exported", manifest
    raw_path = Path(manifest[0]["trace_json"])
    # skill在路径字段保存分号分隔的文件清单；完整CSA应只导出一次launch。
    assert ";" not in str(raw_path), str(raw_path)
    if not raw_path.is_file():
        candidates = list(Path(manifest[0]["export_dir"]).glob("**/simulator/trace.json"))
        assert len(candidates) == 1, candidates
        raw_path = candidates[0]
    simulator = raw_path.parent
    csvs = sorted(simulator.glob("core*/*instr_exe*.csv"), key=lambda p: core_key(p.parent.name))
    expected = {f"core{core}.{kind}" for core in range(24) for kind in ("cubecore0", "veccore0", "veccore1")}
    assert {p.parent.name for p in csvs} == expected and len(csvs) == 72
    assert {p.parent.name for p in simulator.glob("core*/trace.json")} == expected
    statistics = {}
    for path in csvs:
        calls, cycles, instructions, arithmetic = Counter(), Counter(), Counter(), Counter()
        for row in csv.DictReader(path.open()):
            calls[row["pipe"]] += int(row["call_count"])
            cycles[row["pipe"]] += int(row["cycles"])
            instructions[row["instr"]] += int(row["call_count"])
            if row["instr"] not in SYNC:
                arithmetic[row["pipe"]] += int(row["call_count"])
        statistics[path.parent.name] = dict(calls_by_pipe=dict(calls), cycles_sum_by_pipe=dict(cycles),
                                          calls_by_instruction=dict(instructions), nonsync_calls_by_pipe=dict(arithmetic))
        actual_pipe = "CUBE" if ".cubecore" in path.parent.name else "VECTOR"
        assert arithmetic[actual_pipe] > 0, path
        if actual_pipe == "CUBE":
            assert instructions["MMAD"] > 0, path
    print("已核对72份CSV和72条原始核轨迹；读取完整共同时间轴", flush=True)
    raw = json.loads(raw_path.read_text())
    bounds, per_pipe = {}, defaultdict(list)
    computation = defaultdict(list)
    phase_counts = Counter()
    for event in raw["traceEvents"]:
        phase_counts[event.get("ph", "")] += 1
        if event.get("ph") != "X":
            continue
        core = event.get("pid")
        assert core in expected, core
        begin, end = event["ts"], event["ts"] + event.get("dur", 0)
        if core not in bounds:
            bounds[core] = [begin, end]
        else:
            bounds[core] = [min(begin, bounds[core][0]), max(end, bounds[core][1])]
        pipe = event.get("tid", "")
        if event.get("name") not in SYNC:
            per_pipe[core, pipe].append((begin, end))
            if pipe in COMPUTE_PIPES:
                computation[core].append((begin, end))
    assert set(bounds) == expected
    DESTINATION.mkdir(parents=True, exist_ok=True)
    overview, rows, core_rows = [], [], []
    for core in sorted(expected, key=core_key):
        overview += [dict(name="process_name", ph="M", pid=core, args=dict(name=core)),
                     dict(name="process_sort_index", ph="M", pid=core, args=dict(sort_index=core_key(core)[0]*3+core_key(core)[1]))]
        start, finish = bounds[core]
        busy = intervals_union(computation[core])
        gaps = []
        cursor = start
        for begin, end in busy:
            if begin > cursor:
                gaps.append((begin-cursor, cursor, begin))
            cursor = max(cursor, end)
        if cursor < finish:
            gaps.append((finish-cursor, cursor, finish))
        core_rows.append(dict(core=core, start_us=start, end_us=finish, span_us=finish-start,
                              compute_mte_union_us=sum(e-b for b,e in busy),
                              largest_compute_mte_gaps=sorted(gaps, reverse=True)[:12]))
        for pipe, calls in sorted(statistics[core]["calls_by_pipe"].items()):
            spans = intervals_union(per_pipe[core, pipe])
            rows.append(dict(core=core, pipeline=pipe, instruction_calls=calls,
                             instruction_cycles_sum=statistics[core]["cycles_sum_by_pipe"][pipe],
                             nonsync_interval_union_us=sum(e-b for b,e in spans)))
            if pipe in COMPUTE_PIPES:
                overview.append(dict(name="thread_name", ph="M", pid=core, tid=pipe, args=dict(name=pipe)))
                overview.extend(dict(name=pipe, ph="X", pid=core, tid=pipe, ts=b, dur=e-b) for b,e in spans)
    # 小型概览便于同时查看72核；详细指令/同步仍以原始轨迹为准。
    (DESTINATION / "CSA_128K_B16_S6_72cores_model_bounded.pipe_overview.json").write_text(json.dumps(dict(displayTimeUnit="ns",traceEvents=overview)))
    with (DESTINATION / "pipeline_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (DESTINATION / "core_ranges_and_gaps.json").write_text(json.dumps(core_rows,ensure_ascii=False,indent=2)+"\n")
    (DESTINATION / "instruction_counts_by_core.json").write_text(json.dumps(statistics,ensure_ascii=False,indent=2)+"\n")
    report = dict(coverage_passed=True, cores=72, aic=24, aiv=48, source_trace=str(raw_path),
                  run_root=str(run), raw_events=len(raw["traceEvents"]), phases=dict(phase_counts),
                  shared_axis_span_us=max(v[1] for v in bounds.values())-min(v[0] for v in bounds.values()),
                  timestamp_policy="保留完整launch共同时间轴，不逐核归零；us来自原始Chrome事件字段",
                  cycle_policy="cycles_sum是可重叠的指令周期求和，不能当作墙钟耗时或直接相减求收益",
                  gaps_policy="计算/MTE指令区间之外的空隙；可能包含合法Scalar、等待或控制，尚未归因为浪费",
                  output_validation=str(HERE / "REPLAY_VALIDATION.json"), simulator_only=True,
                  bounded_model_backoff=True, backoff_read_limit=128, timing_semantics_changed=True,
                  timing_limit="本地模拟诊断副本；计算/数据/依赖条件不变，轮询退避不同，不能作为正式完整耗时或等待定量证据")
    (HERE / "INCORE_COVERAGE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (DESTINATION / "incore_coverage.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    shutil.copy2(case / "workload_bindings.json", DESTINATION / "workload_bindings.json")
    shutil.copy2(run / "manifest_export.csv", DESTINATION / "manifest_export.csv")
    shutil.copy2(run / "summary.txt", DESTINATION / "skill_summary.txt")
    # 原始数据必须独立保留；同一文件系统硬链接避免重复复制巨量数据，不改原文件。
    import os
    shutil.copytree(simulator, DESTINATION / "raw_simulator", copy_function=os.link, dirs_exist_ok=False)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
