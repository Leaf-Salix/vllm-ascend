# SPDX-License-Identifier: Apache-2.0
"""只按本轮八档六列规范汇总；原始L1校验后取所有有效样本中位数。"""

import csv
import importlib.util
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLUMNS = ("Native", "BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD")


def read(path):
    return json.loads(Path(path).read_text())


def l1_samples(report, side):
    windows = [w for w in report["windows"] if w["side"] == side]
    if len(windows) < 10:
        raise ValueError("不足每侧10个窗口")
    samples = []
    for window in windows:
        raw = read(window["raw"])
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1 and rows
        assert all(len(r) == 6 and r[3] > 0 and r[4] >= r[3] for r in rows)
        if side == "spmd":
            assert len(rows) == len({r[0] for r in rows}) == 72
            assert len({r[1] for r in rows}) == 1
        else:
            assert len({r[1] for r in rows}) > 1
        value = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
        assert abs(value - window["incore_span_us"]) < 1e-6
        samples.append(value)
    return samples, [w["raw"] for w in windows]


def main():
    manifest = read(HERE / "manifest.json")
    cases = read(HERE / "cases.json")["cases"]
    cells = read(HERE / "inventory.json")
    jobs = read(HERE / "jobs.json")
    invalid = []
    for job in jobs:
        status = subprocess.check_output(["task-submit", "--status", job["task_id"]], text=True).strip()
        job["last_observed_status"] = status
        case = next(c for c in cases if c["name"] == job["case"])
        family = job["family"]
        row = cells[case["id"]]
        names = (
            ["Native"]
            if family == "native"
            else (["BSH-MPMD-L1", "BSH-SPMD"] if family == "bsh" else ["TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD"])
        )
        path = Path(job["output"]) / ("report.json" if family == "native" else "result.json")
        if status in ("pending", "running"):
            continue
        try:
            if status != "completed (exit=0)":
                raise ValueError(status)
            report = read(path)
            log_path = Path(job["output"]) / "task.log"
            if not log_path.exists():
                with log_path.open("w") as stream:
                    subprocess.run(["task-submit", "--log", job["task_id"]], stdout=stream, check=True)
            lines = log_path.read_text(errors="replace").splitlines()
            binding = [
                json.loads(line.split("CSA_SHARED_TOOLCHAIN ", 1)[1])
                for line in lines
                if line.startswith("CSA_SHARED_TOOLCHAIN ")
            ]
            assert len(binding) == 1 and binding[0]["status"] == "validated"
            assert binding[0]["release"] == manifest["shared_toolchain"]["release"]
            if family == "native":
                assert report["status"] == "MEASURED" and report["case"] == case
                spec = importlib.util.spec_from_file_location(
                    "native_summary", HERE.parent / "csa_five_way_latest_20261020/summarize.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _, evidence = module.outer(path, report, native=True)
                measured = {"Native": (evidence["samples_us"], [evidence["profile"]["raw_csv"]])}
            else:
                assert report["stage"] == "complete" and report["passed"] and report["case"] == case
                assert all(
                    all(check["guards"].values())
                    and all(value["status"] == "PASS" for value in check["numerical"].values())
                    for check in report["checks"].values()
                )
                label = "BSH" if family == "bsh" else "TND"
                measured = {}
                for side, column in (("mpmd", f"{label}-MPMD-L1"), ("spmd", f"{label}-SPMD")):
                    if report.get("skip_l1_side") == side:
                        assert row[column]["status"] == "PASS", "跳过采集必须已有合格历史样本"
                    else:
                        measured[column] = l1_samples(report, side)
                if family == "tnd":
                    outer = report["outer_profile"]
                    with open(outer["raw_csv"]) as stream:
                        types = Counter(r["Type"] for r in csv.DictReader(stream))
                    assert any("aicpu" in name.lower() for name in types)
                    assert any("aicore" in name.lower() for name in types)
                    assert outer["span_includes_all_streams"] and outer["replays"] == 30
                    measured["TND-MPMD-AICPU"] = (outer["span_us_all"], [outer["raw_csv"]])
            for name, (samples, raw) in measured.items():
                if name in job.get("exclude_columns", []):
                    invalid.append(dict(**job, column=name, reason=job["exclusion_reason"], report=str(path)))
                    continue
                assert len(samples) >= 10 and all(s > 0 for s in samples)
                source_reports = [str(path)]
                source_tasks = [job["task_id"]]
                if name in job.get("merge_columns", []) and row[name]["status"] == "PASS":
                    previous = row[name]
                    samples = previous["samples_us"] + samples
                    raw = previous["raw"] + raw
                    source_reports = previous.get("source_reports", [previous["source_report"]]) + source_reports
                    source_tasks = previous.get("source_tasks", [previous["task_id"]]) + source_tasks
                row[name] = dict(
                    status="PASS",
                    reuse=False,
                    samples_us=samples,
                    valid_windows=len(samples),
                    median_us=statistics.median(samples),
                    task_id=job["task_id"],
                    source_report=str(path),
                    raw=raw,
                    input_contract=case,
                    implementation=family,
                    manifest="manifest.json",
                    correctness="完整结果中的eager/graph/post-L1及保护区检查通过",
                    device=report.get("device_id", report.get("device")),
                    actual_toolchain_selection=binding[0],
                    runtime_environment=report.get("environment"),
                    actual_sources=report.get("sources"),
                    actual_policies=report.get("policies"),
                    task_log=str(log_path),
                    source_date="2026-10-22",
                    source_reports=source_reports,
                    source_tasks=source_tasks,
                    sample_definition="一次完整链路graph replay；无Host复位/编译/warmup时间",
                    sampling=manifest["sampling"],
                    source_mapping="manifest.json中的" + family,
                    raw_source_label_note=(
                        "旧runner的sources.mpmd字段仍写入CLI candidate；BSH实际按family分支调用冻结BSH NativeCSACall"
                        if family == "bsh"
                        else None
                    ),
                )
        except (AssertionError, ValueError, OSError, KeyError) as error:
            invalid.append(dict(**job, reason=str(error), report=str(path)))
            for name in names:
                if row[name]["status"] != "PASS":
                    row[name] = dict(status="FAIL", task_id=job["task_id"], reason=str(error), report=str(path))
    (HERE / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    (HERE / "PERFORMANCE.json").write_text(
        json.dumps(
            dict(
                manifest=manifest,
                cells=cells,
                invalid=invalid,
                native_precision_note="Native保持原数值路径、return_value=False不导出Indexer分数；PTO检查分数。历史Native/PTO非bit差异不改为精确验收。",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    lines = [
        "| 场景 | Native (μs) | BSH-MPMD-L1 (μs) | BSH-SPMD (μs) | TND-MPMD-AICPU (μs) "
        "| TND-MPMD-L1 (μs) | TND-SPMD (μs) | (Native − TND-SPMD) / TND-SPMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        row = cells[case["id"]]
        label = f"{case['history_label'] // 1024}K/B{case['active_requests']}/" + (
            "混长 T56" if case["id"] == "C08" else "S6"
        )
        values = [f"{row[c]['median_us']:.2f}" if row[c]["status"] == "PASS" else row[c]["status"] for c in COLUMNS]
        a, b = row["Native"], row["TND-SPMD"]
        ratio = (
            f"{100 * (a['median_us'] - b['median_us']) / b['median_us']:.2f}%"
            if a["status"] == b["status"] == "PASS"
            else "—"
        )
        lines.append("| " + " | ".join([label, *values, ratio]) + " |")
    (HERE / "PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
