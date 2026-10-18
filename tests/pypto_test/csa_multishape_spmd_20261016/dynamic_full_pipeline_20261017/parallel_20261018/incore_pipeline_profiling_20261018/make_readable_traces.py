#!/usr/bin/env python3
"""把已采集的完整72核图整理成可阅读小包，不运行模型或修改时间。

整体流程：读取已有clean图，保留所有指令和已重建的同步箭头，按核/核组
拆分。长源码栈按PC收录到独立索引，轻量图只留下PC，不复制寄存器参数。
pid/tid只更换为紧凑整数，名称与流水顺序由Chrome元数据明确保留。
原始时间、时长、flow id原样复制；拆分后逐项读回检查，不计算hash。

index.html另用已有pipe区间并集构建离线Canvas总览。它保留72核共同时间轴，
可选择核组、核角色及流水，支持缩放；不把没有计算/MTE事件的位置称为空闲。
ZIP包含总览、24份核组图和源码索引，避免把相同事件的72份单核图再打包一次。
完整原始图及旧清理图保留。全部结果仍来自有界退避的模拟诊断副本。
"""

import json
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from analyze_incore import DESTINATION, core_key
from prepare_standalone import HERE


PREFIX = "CSA_128K_B16_S6_72cores_model_bounded"
ROLE_NAMES = ("AIC", "AIV0", "AIV1")


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    target = DESTINATION / "readable"
    groups_dir, cores_dir = target / "groups", target / "cores"
    groups_dir.mkdir(parents=True, exist_ok=True)
    cores_dir.mkdir(parents=True, exist_ok=True)
    source = DESTINATION / (PREFIX + ".clean.json")
    print("读取已有清理图；不重新采集", flush=True)
    trace = json.loads(source.read_text())
    expected = {f"core{g}.{role}" for g in range(24)
                for role in ("cubecore0", "veccore0", "veccore1")}
    assert {e["pid"] for e in trace["traceEvents"] if e["ph"] == "X"} == expected
    names = sorted(expected, key=core_key)
    core_ids = {name: i for i, name in enumerate(names)}
    tids = {name: {} for name in names}
    for event in trace["traceEvents"]:
        if "tid" in event:
            mapping = tids[event["pid"]]
            mapping.setdefault(event["tid"], len(mapping))
    per_core = defaultdict(list)
    source_stacks = defaultdict(set)
    phase_counts = Counter()
    for event in trace["traceEvents"]:
        core = event["pid"]
        compact = {k: v for k, v in event.items() if k != "args"}
        compact["pid"] = core_ids[core]
        if "tid" in compact:
            compact["tid"] = tids[core][event["tid"]]
        if event["ph"] == "M":
            compact["args"] = dict(event["args"])
            if event["name"] == "process_name":
                group, role = core_key(core)
                compact["args"]["name"] = f"G{group:02d} / {ROLE_NAMES[role]}"
        else:
            args = event.get("args", {})
            pc = args.get("pc_addr")
            if pc:
                compact["args"] = {"pc": pc}
                if args.get("code"):
                    source_stacks[pc].add(args["code"])
        phase_counts[event["ph"]] += 1
        per_core[core].append(compact)
    assert sum(map(len, per_core.values())) == len(trace["traceEvents"])
    # 时间字段沿用原浮点值，不round、不逐核归零、不合并指令。
    shared = {k: trace[k] for k in ("displayTimeUnit", "profilingType", "schemaVersion") if k in trace}
    group_files, core_files = [], []
    checked_counts = Counter()
    for group in range(24):
        group_events = []
        for role in range(3):
            core = names[group * 3 + role]
            events = per_core[core]
            name = f"CSA_G{group:02d}_{ROLE_NAMES[role]}.trace.json"
            path = cores_dir / name
            dump_json(path, dict(shared, traceEvents=events))
            core_files.append(dict(group=group, core=core, file=str(path.relative_to(target)), bytes=path.stat().st_size))
            group_events.extend(events)
        path = groups_dir / f"CSA_G{group:02d}_AIC_AIV0_AIV1.trace.json"
        dump_json(path, dict(shared, traceEvents=group_events))
        # 逐项验证JSON回读；浮点时间/时长和flow id必须完全一致。
        saved = json.loads(path.read_text())["traceEvents"]
        assert saved == group_events
        checked_counts.update(e["ph"] for e in saved)
        group_files.append(dict(group=group, file=str(path.relative_to(target)), bytes=path.stat().st_size,
                                events=len(group_events), core_count=3))
    assert checked_counts == phase_counts
    dump_json(target / "pc_sources.json", {pc: sorted(stacks) for pc, stacks in sorted(source_stacks.items())})
    print("24份核组图、72份单核图完成；全部事件逐项回读一致", flush=True)

    overview = json.loads((DESTINATION / (PREFIX + ".pipe_overview.json")).read_text())
    lanes = {name: defaultdict(list) for name in names}
    for event in overview["traceEvents"]:
        if event["ph"] == "X":
            lanes[event["pid"]][event["tid"]].append([event["ts"], event["dur"]])
    order = ("MTE2", "MTE1", "CUBE", "VECTOR", "FIXP", "FIXPIPE", "MTE3")
    viewer_cores = []
    for name in names:
        group, role = core_key(name)
        viewer_cores.append(dict(name=f"G{group:02d} / {ROLE_NAMES[role]}", group=group, role=role,
                                 pipes={pipe: lanes[name][pipe] for pipe in order if lanes[name].get(pipe)}))
    starts = [row[0] for c in viewer_cores for lane in c["pipes"].values() for row in lane]
    ends = [row[0]+row[1] for c in viewer_cores for lane in c["pipes"].values() for row in lane]
    payload = dict(cores=viewer_cores, groups=group_files, begin=min(starts), end=max(ends))
    template = (HERE / "readable_trace_viewer.html").read_text()
    assert template.count("__TRACE_DATA__") == 1
    html = template.replace("__TRACE_DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"))
    (target / "index.html").write_text(html)
    shutil.copy2(HERE / "READABLE_GUIDE.md", target / "README.md")
    sizes = [r["bytes"] for r in group_files]
    core_sizes = [r["bytes"] for r in core_files]
    report = dict(source=str(source), directory=str(target), cores=72, groups=24,
                  retained_events=sum(phase_counts.values()), event_counts=dict(phase_counts),
                  timestamps_changed=False, instruction_durations_changed=False, sync_flows_changed=False,
                  readback_events_equal=True, overview_interval_union_unchanged=True,
                  removed="指令寄存器参数和重复源码栈；保留PC，源码栈见pc_sources.json。原图保留。",
                  simulator_only=True, bounded_model_backoff=True, timing_semantics_changed=True,
                  backoff_not_calibrated_to_device=True,
                  timing_limit="退避差异可能影响轮询、访存竞争、阶段起点与计算/搬运重叠；结果一致不证明时序或机器指令一致。",
                  html_bytes=(target / "index.html").stat().st_size,
                  group_bytes_min=min(sizes), group_bytes_median=statistics.median(sizes), group_bytes_max=max(sizes),
                  core_bytes_min=min(core_sizes), core_bytes_median=statistics.median(core_sizes), core_bytes_max=max(core_sizes),
                  group_files=group_files, core_files=core_files)
    dump_json(target / "manifest.json", report)
    archive = DESTINATION / "CSA_128K_B16_S6_72cores_readable.zip"
    # ZIP收录全部72核，每核恰好一次；单核版留在目录按需取，不重复装进小包。
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in ("index.html", "README.md", "manifest.json", "pc_sources.json"):
            bundle.write(target / name, f"CSA_readable/{name}")
        for item in group_files:
            bundle.write(target / item["file"], "CSA_readable/" + item["file"])
    report.update(zip=str(archive), zip_bytes=archive.stat().st_size,
                  zip_scope="离线总览 + 24份核组图（全部72核）+ PC源码索引；不重复包含单核版或原始大图")
    (HERE / "READABLE_ARTIFACTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("directory", "html_bytes", "zip_bytes", "group_bytes_min", "group_bytes_max", "core_bytes_min", "core_bytes_max", "retained_events")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
