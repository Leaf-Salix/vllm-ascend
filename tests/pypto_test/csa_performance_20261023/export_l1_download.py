# SPDX-License-Identifier: Apache-2.0
"""从本版八档样本各导出一次真实v22 L1，并保留全部incore任务与物理核。"""

import importlib.util
import json
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MPMD = HERE.parent / "csa_mpmd_aicpu_20261022"
DESTINATION = REPO.parent / "downloads/CSA_TND_MPMD_v22_L1_8cases_20261023"


def read(path):
    return json.loads(Path(path).read_text())


def selected_cpp(raw_path, job):
    """按已冻结A/B的源码差异选v22编排，不按构建目录时间或条形起点猜测。"""
    candidates = list((raw_path.parents[1] / "build_output").glob("decode_csa_tnd_*/orchestration/*.cpp"))
    candidates = [path for path in candidates if "ext_group_plan" in path.read_text()]
    if len(candidates) == 1:
        return candidates[0], "同次运行只有一份TND-MPMD编排"
    kind = job["candidate"]
    if kind == "aicpu_v26_uniform_score_geometry":
        selected = [path for path in candidates if "uniform_six" not in path.read_text()]
        reason = "v26对照任务中选未加入uniform_six的v22控制组"
    elif kind == "aicpu_v22b_bsh_order_live_workers":
        selected = [path for path in candidates if "projection_dependencies" in path.read_text()]
        reason = "v16/v22对照中选带共用投影helper依赖数组的v22候选"
    elif kind == "aicpu_v29_indexer_query_priority":
        selected = []
        for path in candidates:
            section = path.read_text().split("// Spmd qproj_b_spmd: qproj_b")[1].split("rt_submit_aic_task")[0]
            if ".set_dependencies(" not in section:
                selected.append(path)
        reason = "v29对照任务中选Q_B未增加显式优先依赖的v22控制组"
    else:
        raise ValueError(f"尚未定义该配对编排的版本辨识：{kind}")
    assert len(selected) == 1, (kind, selected)
    return selected[0], reason


def main():
    report = read(HERE / "PERFORMANCE.json")
    cases = read(HERE / "cases.json")["cases"]
    jobs = {job["task_id"]: job for job in read(MPMD / "jobs.json")}
    old_path = HERE.parent / "csa_performance_20261022_aicpu/export_l1_download.py"
    spec = importlib.util.spec_from_file_location("csa_l1_names", old_path)
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    DESTINATION.mkdir(parents=True, exist_ok=False)
    index = []
    lines = [
        "# TND-MPMD v22：八档单次完整L1泳道",
        "",
        "对应2026-10-23性能表的TND-MPMD-L1列，算子版本d86c95f69。",
        "每档从全部合格窗口中选最接近该档中位数的一次真实执行，不挑最快值、不拼接多窗。",
        "范围为HC_pre→input RMSNorm→CSA→HC_post；只读已有记录，没有重新上卡。",
        "",
        "拖入Perfetto或chrome://tracing。AIC_00～23、AIV_00～47按真实物理core_id命名。",
        "每个条形使用原始incore begin/end；保留任务间隙、全部实际incore实例和八组输出流水。",
        "Score/Attention标注AIC/AIV，输出投影标注group=0～7；没有合成大阶段条形代替细节。",
        "L1范围外的AICPU活动不在本图中；区间内的调度等待仍保留。",
        "",
        "| 文件 | 本次完整L1 (μs) | 整档中位数 (μs) | 实际task组数 |",
        "|---|---:|---:|---:|",
    ]
    for case in cases:
        cell = report["cells"][case["id"]]["TND-MPMD-L1"]
        selected = min(
            range(len(cell["samples_us"])), key=lambda i: (abs(cell["samples_us"][i] - cell["median_us"]), i)
        )
        raw_path = Path(cell["raw"][selected])
        fragment = next(part for part in cell["fragments"] if str(raw_path) in part["raw"])
        actual = read(fragment["source_report"])
        assert actual["sources"][fragment["side"]] == cell["source"]
        cpp, reason = selected_cpp(raw_path, jobs[fragment["task_id"]])
        _, mapping = mapper.task_map(raw_path, generated_cpp=cpp)
        raw = read(raw_path)
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1
        assert all(len(row) == 6 and 0 <= row[0] < 72 and row[4] >= row[3] > 0 for row in rows)
        assert {row[1] for row in rows} == set(mapping)
        origin = min(row[3] for row in rows)
        factor = 1e6 / raw["metadata"]["clock_freq_hz"]
        span = (max(row[4] for row in rows) - origin) * factor
        assert abs(span - cell["samples_us"][selected]) < 1e-7
        metadata = dict(
            case=case,
            source_implementation="v22 / d86c95f69",
            source_report=fragment["source_report"],
            source_task=fragment["task_id"],
            source_side=fragment["side"],
            source_raw=str(raw_path),
            source_cpp=str(cpp),
            cpp_selection=reason,
            single_invocation=True,
            sample_l1_us=span,
            table_median_us=cell["median_us"],
            qualified_windows=cell["valid_windows"],
            selection="closest real sample to all-qualified-samples median",
            time_boundary="incore begin/end; earliest incore start is zero",
            name_mapping="actual generated orchestration + TMR ring/local TaskId",
            shared_toolchain=report["manifest"]["shared_toolchain"],
        )
        events = []
        for pid, label in ((0, "完整CSA L1"), (1, "AIC"), (2, "AIV")):
            events.extend(
                [
                    dict(ph="M", name="process_name", pid=pid, tid=0, args=dict(name=label)),
                    dict(ph="M", name="process_sort_index", pid=pid, tid=0, args=dict(sort_index=pid)),
                ]
            )
        for core in range(72):
            pid, tid = (1, core) if core < 24 else (2, core - 24)
            label = f"{'AIC' if pid == 1 else 'AIV'}_{tid:02d}"
            events.extend(
                [
                    dict(ph="M", name="thread_name", pid=pid, tid=tid, args=dict(name=label)),
                    dict(ph="M", name="thread_sort_index", pid=pid, tid=tid, args=dict(sort_index=tid)),
                ]
            )
        for core, task, _, start, end, _ in rows:
            info = mapping[task]
            pid, tid = (1, core) if core < 24 else (2, core - 24)
            assert info["kind"] == "task" or info["kind"] == ("aic_task" if pid == 1 else "aiv_task")
            suffix = f" [group={info['group']}]" if "group" in info else ""
            if info["kind"] == "task":
                suffix += f" ({'AIC' if pid == 1 else 'AIV'})"
            events.append(
                dict(
                    ph="X",
                    cat="incore",
                    name=info["name"] + suffix,
                    pid=pid,
                    tid=tid,
                    ts=(start - origin) * factor,
                    dur=(end - start) * factor,
                    args=dict(task_id=task, ring=task >> 32, local_id=task & 0xFFFFFFFF, physical_core_id=core, **info),
                )
            )
        events.append(
            dict(
                ph="X",
                cat="summary",
                name=f"{case['id']} complete CSA L1",
                pid=0,
                tid=0,
                ts=0,
                dur=span,
                args=dict(raw_records=len(rows), task_groups=len(mapping)),
            )
        )
        query = "MixedS1-S6_T56" if case["id"] == "C08" else "S6"
        filename = (
            f"{case['id'][1:]}_{case['history_label'] // 1024}K_B{case['active_requests']}_{query}"
            "_TND-MPMD_v22_L1_SingleCSA.json"
        )
        target = DESTINATION / filename
        target.write_text(
            json.dumps(
                dict(traceEvents=events, displayTimeUnit="us", metadata=metadata),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        # 验证实际保存的文件保留所有原始实例、物理核和时间，而非只检查内存中列表。
        emitted = [event for event in read(target)["traceEvents"] if event.get("cat") == "incore"]
        assert len(emitted) == len(rows)
        assert Counter((event["args"]["physical_core_id"], event["args"]["task_id"]) for event in emitted) == Counter(
            (r[0], r[1]) for r in rows
        )
        for event, row in zip(emitted, rows, strict=True):
            assert abs(event["ts"] - (row[3] - origin) * factor) < 1e-7
            assert abs(event["dur"] - (row[4] - row[3]) * factor) < 1e-7
        assert abs(max(event["ts"] + event["dur"] for event in emitted) - span) < 1e-7
        index.append(
            dict(file=str(target), **metadata, bytes=target.stat().st_size, records=len(rows), task_groups=len(mapping))
        )
        lines.append(f"| {filename} | {span:.2f} | {cell['median_us']:.2f} | {len(mapping)} |")
    lines.extend(
        [
            "",
            "本次单次时间与整档中位数可以不同；不生成虚构的中位数执行。",
            "C05/C08取配对任务中的v22控制组，C07取v16/v22配对中的v22候选；",
            "已按各自真实编排区分版本，均不是未采用的v26/v29或两块query实验。",
            "详细来源同时写在各JSON顶层metadata中。旧20261022下载目录仍保留原版本。",
        ]
    )
    (DESTINATION / "README.md").write_text("\n".join(lines) + "\n")
    archive = DESTINATION.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(DESTINATION.iterdir()):
            bundle.write(path, arcname=f"{DESTINATION.name}/{path.name}")
    with zipfile.ZipFile(archive) as bundle:
        assert len([name for name in bundle.namelist() if name.endswith(".json")]) == 8
        assert bundle.testzip() is None
    receipt = dict(
        directory=str(DESTINATION),
        archive=str(archive),
        archive_bytes=archive.stat().st_size,
        source_report=str(HERE / "PERFORMANCE.json"),
        device_execution=False,
        traces=8,
        all_records_preserved=True,
        cases=index,
    )
    (HERE / "download_manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print({key: receipt[key] for key in ("directory", "archive", "archive_bytes", "traces", "all_records_preserved")})


if __name__ == "__main__":
    main()
