# SPDX-License-Identifier: Apache-2.0
"""从现有完整结果提取CSA与task核内统计；不运行设备、不拼接跨轮收益。"""

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stats(values):
    return dict(
        samples=len(values),
        min_us=min(values),
        mean_us=statistics.mean(values),
        max_us=max(values),
        median_us=statistics.median(values),
    )


def task_name(name):
    name = name.lstrip("_")
    if name.startswith("indexer_score_topk_native_pair"):
        return "score_aic" if "_aic" in name else "score_aiv"
    for prefix in ("qk_pv_aic", "qk_pv_aiv", "proj_b_mm", "qproj_matmul", "qproj_dequant_rms_nope_rope", "proj_a_mm"):
        if name.startswith(prefix):
            return prefix
    return None


def summarize(path):
    report = json.loads(path.read_text())
    summary = {
        "result": str(path.relative_to(HERE)),
        "stage": report["stage"],
        "passed": report.get("passed", False),
        "candidate": report["candidate"],
        "current": report.get("current"),
        "indexer_page_pattern": report.get("indexer_page_pattern", "original"),
        "batch": report["batch"],
        "history": report["history"],
    }
    if not summary["passed"] or "l1" not in report:
        summary["error"] = report.get("error", "").splitlines()[-1:]
        return summary
    l1 = report["l1"]
    summary["full_csa_l1_us"] = l1["statistics_us"]
    if "outer" in l1:
        summary["outer_graph"] = l1["outer"]
    if "boundary_replay" in report:
        summary["boundary_replay"] = {
            "passed": report["boundary_replay"]["passed"],
            "same_graph_replays": report["boundary_replay"].get("same_graph_replays"),
            "result": str(path.parent.relative_to(HERE) / "boundary_replay.json"),
        }
    summary["candidate_reduction_vs_original_pct"] = l1["candidate_reduction_vs_original_pct"]
    summary["candidate_reduction_vs_spmd_pct"] = l1["candidate_reduction_vs_spmd_pct"]
    if "candidate_reduction_vs_current_pct" in l1:
        summary["candidate_reduction_vs_current_pct"] = l1["candidate_reduction_vs_current_pct"]
    summary["incore_scope"] = (
        "每物理核task执行时长，含核内等待/握手；不包含receive_to_start，"
        "不等于纯Cube/Vector运算，也不能把各核时间相加当作关键路径收益"
    )
    timings, per_window = {}, []
    for window in l1["windows"]:
        if window["side"] == "single_spmd":
            continue
        trace = json.loads(Path(window["trace"]).read_text())
        groups = {}
        for event in trace["traceEvents"]:
            name = task_name(event.get("name", ""))
            duration = event.get("args", {}).get("kernel-duration-us")
            if event.get("ph") != "X" or name is None or duration is None:
                continue
            groups.setdefault(name, []).append(duration)
            timings.setdefault(window["side"], {}).setdefault(name, []).append(duration)
        per_window.append(
            dict(side=window["side"], trace=window["trace"], tasks={name: stats(v) for name, v in groups.items()})
        )
    summary["task_incore_us"] = {
        side: {name: stats(v) for name, v in groups.items()} for side, groups in timings.items()
    }
    summary["task_windows"] = per_window
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.results or sorted((HERE / "runs").glob("*/h*/result.json"))
    results = [summarize(path.resolve()) for path in paths]
    destination = HERE / "RESULTS.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("task_windows", "task_incore_us", "incore_scope")},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
