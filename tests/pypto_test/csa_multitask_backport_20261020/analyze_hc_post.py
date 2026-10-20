# SPDX-License-Identifier: Apache-2.0
"""从现有七档泳道分离输出尾段核内执行、启动离散和依赖结束后的间隙。"""

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = [(131072, 4), (131072, 8), (131072, 16), (131072, 24), (131072, 28), (8192, 16), (8192, 32)]


def stats(values):
    return dict(min_us=min(values), mean_us=statistics.mean(values), max_us=max(values))


def analyze(trace_path):
    events = [
        e
        for e in json.loads(trace_path.read_text())["traceEvents"]
        if e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})
    ]
    post = [e for e in events if e["name"].lstrip("_").startswith("proj_b_act_hc_post")]
    proj_b = [e for e in events if e["name"].lstrip("_").startswith("proj_b_mm")]
    if not post or not proj_b:
        raise ValueError(f"缺少独立O-B/融合HCpost任务：{trace_path}")

    def end(e):
        return e["ts"] + e["dur"]

    def begin(e):
        return end(e) - e["args"]["kernel-duration-us"]

    origin = min(map(begin, events))
    first, last = min(map(begin, post)), max(map(end, post))
    producer_end = max(map(end, proj_b))
    return dict(
        trace=str(trace_path),
        full_csa_l1_us=max(map(end, events)) - origin,
        physical_aiv_count=len({(e["pid"], e["tid"]) for e in post}),
        record_count=len(post),
        task_ids=sorted({e["args"]["taskId"] for e in post}),
        kernel_us=stats([e["args"]["kernel-duration-us"] for e in post]),
        local_setup_us=stats([e["dur"] - e["args"]["kernel-duration-us"] for e in post]),
        group_span_us=last - first,
        start_spread_us=max(map(begin, post)) - first,
        gap_after_last_ob_us=first - producer_end,
        tail_after_last_ob_us=last - producer_end,
        first_start_us=first - origin,
        last_end_us=last - origin,
        is_last_task=abs(last - max(map(end, events))) < 1e-6,
    )


def analyze_result(path):
    result = json.loads(path.read_text())
    if not result.get("passed"):
        raise ValueError(f"仅分析完整通过的测量：{path}")
    sides = {}
    for window in result["l1"]["windows"]:
        if window["side"] != "single_spmd":
            sides.setdefault(window["side"], []).append(analyze(Path(window["trace"])))
    summaries = {}
    for side, windows in sides.items():
        summaries[side] = {
            key: stats([row[key] for row in windows])
            for key in (
                "full_csa_l1_us",
                "group_span_us",
                "start_spread_us",
                "gap_after_last_ob_us",
                "tail_after_last_ob_us",
            )
        }
        summaries[side]["physical_aiv_counts"] = sorted({row["physical_aiv_count"] for row in windows})
        summaries[side]["per_window_kernel_mean_us"] = stats([row["kernel_us"]["mean_us"] for row in windows])
    return dict(result=str(path), windows=sides, summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.result:
        report = analyze_result(args.result)
    else:
        download = HERE / "download/CSA_MultiTask_7Cases_L1_L4_20261021"
        manifest = json.loads((download / "manifest.json").read_text())
        rows = []
        for number, (history, batch) in enumerate(CASES, 1):
            label = f"{history // 1024}K_B{batch:02d}"
            levels = {}
            for level in (1, 4):
                path = download / f"{number:02d}_{label}_L{level}_SingleCSA.json"
                levels[f"L{level}"] = analyze(path)
            latest = HERE / "runs/topk_prefix_v1" / f"h{history}_b{batch}_topk_v1/result.json"
            latest_report = analyze_result(latest)
            rows.append(
                dict(
                    history=history,
                    batch=batch,
                    tokens=batch * 6,
                    current_rows_per_worker=2 if batch == 16 else 4,
                    current_sync_start=batch == 16,
                    download=levels,
                    latest_topk_result=str(latest),
                    latest_topk_windows=latest_report["windows"]["multitask_candidate"],
                    latest_topk_summary=latest_report["summaries"]["multitask_candidate"],
                )
            )
        report = dict(
            scope="proj_b_act_hc_post融合八组INT32反量化/规约与HCpost，不能与Native纯HCpost直接比时长",
            timing="incore起点=ts+dur-kernel-duration-us；local setup独列；O-B末尾到post首启仅是间隙，不归因纯AICPU",
            download_formal=manifest["formal_operator_commit"],
            latest_formal="Top-K / sources/topk_prefix_v1",
            warning="下载单窗与Top-K五窗属不同轮/工具链，不计算跨轮收益；L4含插桩，只作调度诊断",
            rows=rows,
        )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.result:
        print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))
    else:
        for row in report["rows"]:
            l1 = row["download"]["L1"]
            print(
                row["history"],
                row["batch"],
                l1["physical_aiv_count"],
                "kernel",
                l1["kernel_us"],
                "gap",
                round(l1["gap_after_last_ob_us"], 3),
                "latest group",
                row["latest_topk_summary"]["group_span_us"],
            )


if __name__ == "__main__":
    main()
