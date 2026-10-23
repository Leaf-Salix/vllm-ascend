# SPDX-License-Identifier: Apache-2.0
"""归档已完成的Q_B分档实验；只读历史结果，不启动设备或编译。"""

import collections
import json
import shutil
import statistics
from pathlib import Path

from summarize_native_output import collect

HERE = Path(__file__).resolve().parent


def stage_statistics(run, side):
    """Worker View计数一次；核占用和整层首尾分别保留，不混成收益。"""
    samples = collections.defaultdict(list)
    for source in sorted(run.glob(f"l1_*_{side}/swimlane.json")):
        events = [e for e in json.loads(source.read_text())["traceEvents"]
                  if e.get("ph") == "X" and e.get("pid") == 4]
        begin = min(e["ts"] for e in events)
        for prefix, aliases in (
            ("hca_qb_stream", ("hca_qb_stream",)),
            ("qproj_dequant", ("qproj_dequant",)),
            ("hca_unified_attention_aic", ("hca_unified_attention_aic",)),
            ("proj_a_mm", ("proj_a_mm", "hca_oa_tall", "hca_native_oa_all")),
            ("hca_native_ob_split", ("hca_native_ob_split",)),
        ):
            selected = [e for e in events if e["name"].startswith(aliases)]
            if not selected:
                continue
            samples[prefix].append({
                "source": str(source.relative_to(HERE)),
                "begin_us": min(e["ts"] for e in selected) - begin,
                "end_us": max(e["ts"] + e["dur"] for e in selected) - begin,
                "mean_kernel_duration_us": statistics.mean(e["args"]["kernel-duration-us"] for e in selected),
                "records": len(selected),
            })
    return {name: {
        "medians": {key: statistics.median(row[key] for row in rows)
                    for key in ("begin_us", "end_us", "mean_kernel_duration_us", "records")},
        "samples": rows,
    } for name, rows in samples.items()}


def main():
    rows = collect("Q_PROJECTION_TASKS.json")
    local_paths = [HERE / "experiments/q_tall_v1", HERE / "runs/q_tall_cpu_v1"]
    for row in rows:
        run = HERE / "runs" / row["name"]
        local_paths.append(run)
        source = run / "result.json"
        archive = HERE / "evidence" / (row["name"] + "_result.json")
        shutil.copy2(source, archive)
        row["archived_result"] = str(archive.relative_to(HERE))
        pair = row["paired_l1"]
        row["candidate_faster_pairs"] = sum(a > b for a, b in zip(
            pair["control"]["samples_us"], pair["candidate"]["samples_us"]))
        row["stages"] = {side: stage_statistics(run, side) for side in ("control", "candidate")}
    shutil.copy2(HERE / "runs/q_tall_cpu_v1/result.json", HERE / "evidence/q_tall_cpu_v1_result.json")
    result = {
        "decision": "不采用；本轮完成后按用户要求暂停，不启动下一轮",
        "metric": "完整HCA直接执行的L1 incore首尾；没有ACLGraph replay",
        "production_changed": False,
        "strict_native_storage_pass": False,
        "rows": rows,
    }
    (HERE / "Q_PROJECTION_SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    artifacts = []
    for path in local_paths:
        files = [p for p in path.rglob("*") if p.is_file()]
        artifacts.append({"path": str(path), "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
    (HERE / "Q_PROJECTION_ARTIFACTS.json").write_text(json.dumps({
        "policy": "冻结源码、编译产物和原始泳道图本地保留；只记录路径/数量/大小，不做hash校验",
        "local": artifacts,
    }, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(row["name"], {k: v["median_us"] for k, v in row["paired_l1"].items()},
              "faster_pairs", row["candidate_faster_pairs"])


if __name__ == "__main__":
    main()
