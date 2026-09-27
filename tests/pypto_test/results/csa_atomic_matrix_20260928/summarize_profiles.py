"""只读本轮模型分项和七档 DFX，记录与历史上游的范围差异。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def main():
    model = json.loads((ROOT / "model/model_gap_rank0.json").read_text())
    lines = ["# 固定规约七档：模型 CSA 与下游差距", "", model["scope"], "", model["limits"], "",
             "正式结论使用[无profiler十步forward](RESULTS.md)，本表不能精确分账另一轮正式计时。", "",
             "CSA每档为rank0三步×21层的63个完整区间，单位μs；其余列为每step总量，单位ms。", "",
             "| 档位 | Native/PTO CSA均值 | Native/PTO CSA P95 | CSA变化 | Native/PTO FFN | 专家GMM增量 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in model["cases"]:
        n, p = row["native"], row["pto"]
        nc, pc = n["csa"], p["csa"]
        gmm = {side: sum(value for name, value in row[side]["ffn_task_busy_us_per_step"].items()
                         if "GroupedMatmul" in name) for side in ("native", "pto")}
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {nc['mean_us']:.2f}/{pc['mean_us']:.2f} | "
                     f"{nc['p95_us']:.2f}/{pc['p95_us']:.2f} | {(pc['mean_us']/nc['mean_us']-1)*100:+.2f}% | "
                     f"{n['phases']['ffn_sum_us']/1000:.3f}/{p['phases']['ffn_sum_us']/1000:.3f} | "
                     f"{(gmm['pto']-gmm['native'])/1000:+.3f} |")
    lines += ["", "专家GMM为两类GroupedMatmul任务duration之和，不是独占关键路径；"
              "FFN含到达等待与通信。不同event模式会影响profile，不能仅据此断言专家数量或硬件效率改变。",
              "当前未保存实际路由索引，固定规约同时改变split-K及舍入分组；不能单独归因为atomic硬件抖动。"]
    (ROOT / "model/MODEL_GAP.md").write_text("\n".join(lines) + "\n")

    path = ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py"
    spec = importlib.util.spec_from_file_location("worker_gap", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    upstream = worker.summarize(worker.UPSTREAM)
    current = []
    for history, batch in CASES:
        report_path = ROOT / "swimlanes" / f"h{history}_b{batch}" / "report.json"
        report = json.loads(report_path.read_text())
        current.append({"history": history, "batch": batch,
                        **worker.summarize(Path(report["swimlane_windows"][0]["merged_swimlane"]))})
    result = {"operator": "71153bb3/atomic0", "source_reference": "pypto-lib main2164563",
              "scope": "本轮每档一个单卡layer4 DFX窗口；正式权重、合成历史、metadata reuse、graph replay。",
              "limits": "历史上游727.98μs图缺源码和完整配置，非同输入A/B；setup并非纯调度耗时，核内含等待。",
              "upstream": upstream, "current": current}
    (ROOT / "worker_gap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 当前固定规约 DFX 与上游参考", "", result["scope"], "", result["limits"], "",
             "## 七档窗口", "", "单位μs；单窗口，不是无profiler均值或P95，不替代整模型750μs验收。", "",
             "| 档位 | Worker首尾 | Score AIC核内均值 | Score AIV核内均值 | Top-K merge核内均值 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in current:
        tasks = row["tasks"]
        values = [tasks[name]["kernel_mean_us"] for name in (
            "indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge")]
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {row['worker_span_us']:.2f} | "
                     + " | ".join(f"{value:.2f}" for value in values) + " |")
    short = next(row for row in current if row["history"] == 8192 and row["batch"] == 16)
    lines += ["", "## 8K/B16 与历史上游727.98μs图", "",
              "| 分段 μs | 历史上游 | 当前 |", "| --- | ---: | ---: |"]
    labels = ("首Worker→norm结束", "norm结束→Sparse首receive", "Sparse首receive→merge结束", "merge结束→末Worker")
    for label, key in zip(labels, upstream["phases_us"]):
        lines.append(f"| {label} | {upstream['phases_us'][key]:.2f} | {short['phases_us'][key]:.2f} |")
    lines += ["", "| Task | 上游/当前数量 | 上游/当前核内均值 μs | 上游/当前启动分散 μs | 上游/当前setup均值 μs |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in dict.fromkeys([*upstream["tasks"], *short["tasks"]]):
        a, b = upstream["tasks"].get(name), short["tasks"].get(name)
        values = []
        for key in ("blocks", "kernel_mean_us", "start_spread_us", "setup_mean_us"):
            values.append("/".join(f"{value[key]:.2f}" if value else "—" for value in (a, b)))
        lines.append("| " + " | ".join([name, *values]) + " |")
    lines += ["", "## 为何与pypto-lib模式不同", "",
              "源码参考2164563的QA/KV分别split-K 2/4，使用atomic Add且需要GM种子清零。"
              "当前部署选择1/1固定归约，依据是真实EP16七档forward和DSpark共同通过；"
              "不是只追求单独CSA的最高并行度。上游独立用例不能替代本模型后续MoE工作量验收。", "",
              "当前仍复用Native NZ权重和原生物理cache页；PTO内部页表/slot读取、Native metadata/状态接口"
              "以及长短历史的query分组均以真实接入为约束。"
              "上游连续cache模式和历史图配置不能直接假定与这些额外工作等价。", "",
              "固定规约下旧清零已成为冗余；独立免seed候选单卡通过，尚待两档EP16。"
              "此处七档图和整网成绩均不含该候选。全部逐任务数据见[worker_gap.json](worker_gap.json)。"]
    (ROOT / "WORKER_GAP.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
