# SPDX-License-Identifier: Apache-2.0
"""仅汇总指定冻结候选的一轮七档；逐域等权后按128K:8K=8:2计收益。"""

import argparse
import json
import statistics
from pathlib import Path

from summarize import HERE, summarize

CASES = [(131072, b) for b in (4, 8, 16, 24, 28)] + [(8192, b) for b in (16, 32)]


def build(candidate, attempt):
    rows, environments = [], []
    for history, batch in CASES:
        path = HERE / "runs" / candidate / f"h{history}_b{batch}_{attempt}" / "result.json"
        raw = json.loads(path.read_text())
        if not raw["passed"] or raw["stage"] != "complete":
            raise ValueError(f"七档尚未完整通过：{path}")
        if (raw["history"], raw["batch"], raw["query_tokens"], raw["candidate"]) != (
            history,
            batch,
            6,
            candidate,
        ):
            raise ValueError(f"结果描述符与档位不符：{path}")
        if raw.get("indexer_page_pattern", "original") != "original":
            raise ValueError("不能将扩展页布局混入七档性能表")
        summarized = summarize(path)
        stats = raw["l1"]["statistics_us"]
        if any(len(v["samples_us"]) != 5 for v in stats.values()):
            raise ValueError("七档主口径必须保留整组五窗")
        environments.append(raw["environment"])
        topology = {}
        for role in ("multitask_candidate", "single_spmd"):
            window = next(w for w in raw["l1"]["windows"] if w["side"] == role)
            trace = json.loads(Path(window["trace"]).read_text())
            ids = {
                e["args"]["taskId"]
                for e in trace["traceEvents"]
                if e.get("ph") == "X" and "taskId" in e.get("args", {})
            }
            topology[role] = len(ids)
        if topology["multitask_candidate"] <= 1 or topology["single_spmd"] != 1:
            raise ValueError("任务拓扑不符合多task对单SPMD的合同")
        before, now, reference = (
            stats[k]["mean_us"] for k in ("multitask_original", "multitask_candidate", "single_spmd")
        )
        rows.append(
            dict(
                history=history,
                batch=batch,
                result=str(path.relative_to(HERE)),
                l1_us=stats,
                outer=raw["l1"].get("outer"),
                active_task_ids=topology,
                incore_us=summarized["task_incore_us"],
                reduction_vs_original_pct=100 * (before - now) / before,
                reduction_vs_spmd_pct=100 * (reference - now) / reference,
                boundary_replay=raw.get("boundary_replay", {}).get("passed"),
                scope=raw["scope"],
            )
        )
    if any(e != environments[0] for e in environments[1:]):
        raise ValueError("七档环境不一致，不能按同一冻结环境汇总")
    regions = {}
    for history in (131072, 8192):
        subset = [r for r in rows if r["history"] == history]
        regions[str(history)] = {
            key: statistics.mean(r[key] for r in subset)
            for key in ("reduction_vs_original_pct", "reduction_vs_spmd_pct")
        }
    return dict(
        candidate=candidate,
        attempt=attempt,
        passed=True,
        environment=environments[0],
        scope="同轮/同卡/同根输入；L1最早incore到最晚结束，外层单列；非整模型",
        weighting="各history内对每档百分比等权平均，再128K乘0.8、8K乘0.2",
        rows=rows,
        regions=regions,
        weighted={key: 0.8 * regions["131072"][key] + 0.2 * regions["8192"][key] for key in regions["131072"]},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--attempt", default="matrix_v1")
    args = parser.parse_args()
    if not args.candidate.isidentifier() or not args.attempt.isidentifier():
        raise ValueError("候选和attempt必须是标识符")
    report = build(args.candidate, args.attempt)
    destination = HERE / f"MATRIX_{args.candidate}_{args.attempt}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(destination)
    print(json.dumps(report["weighted"], ensure_ascii=False))


if __name__ == "__main__":
    main()
