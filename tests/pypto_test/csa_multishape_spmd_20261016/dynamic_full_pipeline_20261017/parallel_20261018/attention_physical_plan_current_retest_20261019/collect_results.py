# SPDX-License-Identifier: Apache-2.0
"""仅汇总已存在的任务结果和实际编译器记录；不提交任务、不重新测量。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def collect(registry):
    rows = []
    for entry in read(HERE / registry):
        folder = Path(entry["output"])
        report_path, matrix_path = folder / "result.json", folder / "source_matrix/result.json"
        if not report_path.exists() or not matrix_path.exists():
            continue
        report, matrix = read(report_path), read(matrix_path)
        if not report.get("passed") or not matrix.get("passed"):
            continue
        rows.append({
            **{k: v for k, v in entry.items() if k in ("batch", "history", "task", "variant", "exit_code", "terminal_status")},
            "passed": True,
            "statistics_us": matrix["statistics_us"],
            "candidate_vs_reference": matrix["candidate_vs_reference"],
            "comparison": matrix["comparison"],
            "metadata_replay_passed": report.get("metadata_replay", {}).get("passed"),
            "padding_replay_passed": report.get("padding_replay", {}).get("passed"),
            "post_l1_protocol": report.get("post_l1_protocol"),
            "graph_replay_validated": report.get("graph_replay_validated"),
            "policies": matrix["policies"],
            "kernel_configs": matrix["kernel_configs"],
            "full_report": str(report_path),
            "matrix_report": str(matrix_path),
        })
    return rows


for registry, output in (("DEVICE_SUBMISSIONS.json", "COMPLETED_RESULTS.json"),
                         ("BOUNDARY_SUBMISSIONS.json", "BOUNDARY_RESULTS.json")):
    rows = collect(registry)
    (HERE / output).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(registry, row["history"], row["batch"], row.get("variant", "v1"),
              {k: v["median_us"] for k, v in row["statistics_us"].items()})
