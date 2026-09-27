#!/usr/bin/env python3
"""汇总本轮单卡矩阵；只读取已有结果，不触发设备测试。"""

import argparse
import json
import shutil
import statistics
from pathlib import Path

CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def read_json(path):
    return json.loads(path.read_text())


def stats(value):
    if value.get("eliminated"):
        # No independent operation remains; do not present zero placeholders
        # from older diagnostic writers as measured empty-graph samples.
        return {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "samples": 0, "eliminated": True}
    return {
        "mean_us": statistics.mean(value["samples_us"]),
        "p50_us": value["us_p50"],
        "p95_us": value["us_p95"],
        "samples": len(value["samples_us"]),
    }


def guard_failures(value, prefix=""):
    failures = []
    if isinstance(value, dict):
        if "status" in value and value["status"] != "PASS":
            failures.append(prefix + ":" + value["status"])
        for key, child in value.items():
            failures.extend(guard_failures(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(guard_failures(child, f"{prefix}[{index}]"))
    return failures


def summarize_timing(path):
    report = read_json(path)
    timing = report["timing"]
    phases = timing.get("split_cache_phases")
    if phases is None:
        if report.get("indexer_cache_binding", {}).get("history_copy_before_csa") is not False:
            raise ValueError(f"Missing cache phase boundary: {path}")
        phases = {"csa_body": timing["pto"], "split": {"eliminated": True},
                  "slot_writeback": {"eliminated": True}}
    output = report["pto_native"]["x_out"]
    guards = {key: report[key] for key in ("native_guards", "pto_guards")}
    for side in ("native", "pto"):
        guards[side + "_graph"] = timing[side]["guards"]
    return {
        "report": str(path),
        "status": report["status"],
        "native": stats(timing["native"]),
        "pto_full": stats(timing["pto"]),
        "body": stats(phases["csa_body"]),
        "split": stats(phases["split"]),
        "writeback": stats(phases["slot_writeback"]),
        "guard_failures": guard_failures(guards),
        "output": {key: output[key] for key in ("status", "nonfinite", "max_abs", "rmse")},
        "nonfinite_by_tensor": {
            key: value["nonfinite"] for key, value in report["pto_native"].items() if value.get("nonfinite")
        },
        "topk": {
            key: report["topk_selection"][key] for key in ("replaced_indices", "invalid_rows", "structural_errors")
        },
    }


def summarize_swimlane(path):
    events = read_json(path)["traceEvents"]
    worker_pids = {
        e["pid"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"
    }
    score = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("pid") in worker_pids
        and "indexer_score_topk" in e.get("name", "")
        and "aic_spmd" in e["name"]
    ]
    if not score:
        raise ValueError(f"缺少 Score Worker 事件: {path}")
    return {
        "path": str(path),
        "score_kernel": score[0]["name"].split("(")[0],
        "blocks": len(score),
        "aic_count": len({e["tid"] for e in score}),
        "incore_mean_us": statistics.mean(e["args"]["kernel-duration-us"] for e in score),
        "worker_span_us": max(e["ts"] + e["dur"] for e in score) - min(e["ts"] for e in score),
        "score_to_publish_us": max(
            e["ts"] + e["dur"]
            for e in events
            if e.get("ph") == "X" and e.get("pid") in worker_pids and "indexer_topk_" in e.get("name", "")
        )
        - min(e["ts"] for e in score),
    }


def collect(root, copy_artifacts):
    previous = root.parent / "csa_split_optimization_20260927"
    rows = []
    artifacts = []
    for history, batch in CASES:
        reused = (history, batch) == (131072, 16)
        row = {"history": history, "batch": batch, "reused_previous": reused}
        for version in ("v4", "v7"):
            label = {"v4": "v4_buffered_score", "v7": "v7_native_overlap"}[version]
            case = (previous / label if reused else root / version) / f"h{history}_b{batch}"
            report_path = case / "timing/report.json"
            row[version] = summarize_timing(report_path) if report_path.exists() else {"status": "PENDING"}
            row[version]["pypto_revision"] = "2a4e09ff" if reused and version == "v4" else "3e87a843"
            if version != "v7":
                continue
            row[version]["swimlane"] = [
                summarize_swimlane(path) for path in sorted((case / "swimlane/dfx").glob("**/merged_swimlane.json"))
            ]
            prefix = f"h{history}_b{batch}"
            for side in ("native", "pto"):
                traces = list((case / f"timing/profile/{side}").glob("**/trace_view.json"))
                if len(traces) > 1:
                    raise ValueError(f"不应混合多次采集: {case}, {side}")
                for trace in traces:
                    artifacts.append((trace, f"{prefix}_{side}_pytorch.json"))
                    if side == "native":
                        trace_data = read_json(trace)
                        trace_events = trace_data if isinstance(trace_data, list) else trace_data["traceEvents"]
                        row[version]["profile_native_qli_us"] = [
                            event["dur"]
                            for event in trace_events
                            if event.get("ph") == "X" and event.get("name") == "VllmQuantLightningIndexer"
                        ]
            for index, swimlane in enumerate(row[version]["swimlane"]):
                artifacts.append((Path(swimlane["path"]), f"{prefix}_pto_swimlane_w{index}.json"))
        if all("body" in row[v] for v in ("v4", "v7")):
            row["body_change_pct"] = 100 * (row["v7"]["body"]["mean_us"] / row["v4"]["body"]["mean_us"] - 1)
            row["body_vs_native_pct"] = 100 * (row["v7"]["body"]["mean_us"] / row["v7"]["native"]["mean_us"] - 1)
        rows.append(row)
    if copy_artifacts:
        bundle = root / "download"
        bundle.mkdir(exist_ok=True)
        for source, name in artifacts:
            shutil.copy2(source, bundle / name)
        (bundle / "manifest.json").write_text(
            json.dumps([{"file": name, "source": str(source)} for source, name in artifacts], indent=2) + "\n"
        )
    return rows


def render(rows):
    lines = [
        "# Native Cube Score：七档单卡对照（2026-09-27）",
        "",
        "正式 layer 4 权重、合成输入与历史；S=6、TP=1、mode=2、atomic=1、确定性 level=0、EPLB 关闭。",
        "按第二个 CSA 层复用 compact metadata；每条计时路径预热5次、无 profiler 采样20次。",
        "本体含 HC_pre → norm → CSA → HC_post，排除入口 cache 拆分和出口 slot 写回。",
        "完整 PTO 路径包含拆分和写回；各阶段单独捕图计时，阶段均值之和不要求等于完整路径。",
        "",
        "六个补测档位的 v4/v7 共用 PyPTO 3e87a843、Simpler a54c05095、PTOAS 0.66、PTO-ISA 327cd586。",
        "v4 源码 da6f474a，v7 源码 9516acbe。"
        "128K/B16 复用前轮记录，其 v4 使用 PyPTO 2a4e09ff，不能称为严格同工具链 A/B。",
        "8K 仍走原短上下文 Score 路径，v7 新 Cube 路径只覆盖本矩阵的 128K 档位。",
        "",
        "单位 μs，变化为 v7/v4−1；负数表示降低。本轮是单卡测量，不是16卡整模型验收。",
        "",
        "| H / B | v4同轮Native | v7同轮Native | v4本体均值 | v7本体均值 | 变化 | v7对Native | v7本体p50 / p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "body_change_pct" not in row:
            continue
        a, b = row["v4"], row["v7"]
        label = f"{row['history'] // 1024}K / {row['batch']}" + ("（复用）" if row["reused_previous"] else "")
        lines.append(
            f"| {label} | {a['native']['mean_us']:.2f} | {b['native']['mean_us']:.2f} | "
            f"{a['body']['mean_us']:.2f} | {b['body']['mean_us']:.2f} | {row['body_change_pct']:+.2f}% | "
            f"{row['body_vs_native_pct']:+.2f}% | {b['body']['p50_us']:.2f} / {b['body']['p95_us']:.2f} |"
        )
    lines += [
        "",
        "| H / B | v4完整PTO | v7完整PTO | v7拆分 | v7本体 | v7写回 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "body_change_pct" not in row:
            continue
        a, b = row["v4"], row["v7"]
        lines.append(
            f"| {row['history'] // 1024}K / {row['batch']} | {a['pto_full']['mean_us']:.2f} | "
            f"{b['pto_full']['mean_us']:.2f} | {b['split']['mean_us']:.2f} | {b['body']['mean_us']:.2f} | "
            f"{b['writeback']['mean_us']:.2f} |"
        )
    lines += [
        "",
        "浮点列为对 Native 的零容差差异诊断，不把 FAIL 改写为精度验收通过；完整误差、guard和索引结构见 summary.json。",
        "",
        "| H / B | v4输出RMSE | v7输出RMSE | v7输出max_abs | v4/v7 Top-K集合替换 | v7 guard失败数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "body_change_pct" not in row:
            continue
        a, b = row["v4"], row["v7"]
        lines.append(
            f"| {row['history'] // 1024}K / {row['batch']} | {a['output']['rmse']:.7f} | "
            f"{b['output']['rmse']:.7f} | {b['output']['max_abs']:.7f} | "
            f"{a['topk']['replaced_indices']} / {b['topk']['replaced_indices']} | "
            f"{len(b['guard_failures'])} |"
        )
    lines += [
        "",
        "Indexer 单独观察：Native 列为 PyTorch trace 的 QLI kernel，"
        "PTO 列为四次独立 DFX 的 Score 首 Worker→Top-K 发布末 Worker。",
        "两侧独立采集；PTO 窗口含任务调度及与其他分支交叠，未包含之前的系数准备，"
        "不等于独占算术时间，也不代替无 profiler 主计时。",
        "",
        "| H / B | Native QLI μs | PTO Score→publish μs（四窗口范围） | PTO Score核内 μs（四窗口均值范围） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        b = row["v7"]
        windows = b.get("swimlane", [])
        if not windows:
            continue
        native = ", ".join(f"{v:.2f}" for v in b.get("profile_native_qli_us", [])) or "未补采"
        span = [w["score_to_publish_us"] for w in windows]
        incore = [w["incore_mean_us"] for w in windows]
        lines.append(
            f"| {row['history'] // 1024}K / {row['batch']} | {native} | {min(span):.2f}～{max(span):.2f} | "
            f"{min(incore):.2f}～{max(incore):.2f} |"
        )
    lines += [
        "",
        "六个补测档位各保存 Native/PTO PyTorch trace 和 v7 四窗口 PTO 泳道。"
        "128K/B16 只复用已有四窗口泳道，未补采 PyTorch trace。",
        "[统一下载目录](download/)；[逐项汇总与原始报告路径](summary.json)。下载目录中的 manifest.json 记录来源。",
        "PyTorch trace 覆盖完整路径，PTO 泳道覆盖本体，两者均为独立采集；"
        "DFX只统计Worker View，不与Scheduler View叠加。",
        "",
        "复现：在工作区准备源码 da6f474a / 9516acbe 对应冻结 worktree（路径见 run_case.sh），共用当前 Native 扩展；",
        "通过 `task-submit --device 0 --max-time 2400 'bash <本目录>/run_pair.sh <history> <batch>'` 执行。",
        "`python <本目录>/summarize.py --copy-artifacts` 只汇总已有结果并生成下载目录。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-artifacts", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows = collect(root, args.copy_artifacts)
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    report_text = render(rows)
    (root / "README.md").write_text(report_text)
    if args.copy_artifacts:
        (root / "download/README.md").write_text(report_text.replace("(download/)", "(./)"))
        shutil.copy2(root / "summary.json", root / "download/summary.json")
    for row in rows:
        print(row["history"], row["batch"], row.get("body_change_pct", "PENDING"))


if __name__ == "__main__":
    main()
