"""呈现已采模型区间；完整CSA与分段相加所用本体口径分别标明。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "model"


def main():
    data = json.loads((ROOT / "model_gap_rank0.json").read_text())
    lines = [
        "# 自适应KV：真实EP16模型内区间", "", data["scope"], "", data["limits"], "",
        "正式验收使用[无profiler十步forward](RESULTS.md)。本表是另一轮rank0独立profile，不能精确分账正式耗时。",
        "每档63个完整CSA区间（3步×21层），包含首次metadata。仅统计rank0，不替代全rank验收。", "",
        "| 档位 | Native/PTO CSA均值 μs | Native/PTO CSA P50 μs | Native/PTO CSA P95 μs | 每步完整CSA合计 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in data["cases"]:
        n, p = (row[side]["csa"] for side in ("native", "pto"))
        if n["samples"] != 63 or p["samples"] != 63:
            raise ValueError("本报告要求两侧均为63个CSA区间")
        lines.append(
            f"| {row['history']//1024}K/B{row['batch']} | {n['mean_us']:.2f}/{p['mean_us']:.2f} | "
            f"{n['p50_us']:.2f}/{p['p50_us']:.2f} | {n['p95_us']:.2f}/{p['p95_us']:.2f} | "
            f"{n['mean_us']*21/1000:.3f}/{p['mean_us']*21/1000:.3f} |")
    lines += [
        "", "以下分段均为每步均值，单位ms。CSA本体不含根调用之前的metadata，首次metadata计入区间外；",
        "因此这四个互斥分段可相加为主图区间。不要把上表完整CSA合计再与区间外相加。", "",
        "| 档位 | 侧 | CSA本体合计 | 其他Attention合计 | FFN合计 | 半层区间外 | 主图区间 | 专家GMM任务合计 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["cases"]:
        for side in ("native", "pto"):
            value = row[side]
            phase = value["phases"]
            totals = [phase[key] / 1000 for key in (
                "c4_body_sum_us", "other_attention_sum_us", "ffn_sum_us",
                "outside_half_intervals_us", "main_graph_us")]
            if abs(sum(totals[:4]) - totals[4]) > 0.001:
                raise ValueError("互斥分段未覆盖主图区间")
            totals.append(sum(v for k, v in value["ffn_task_busy_us_per_step"].items()
                              if "GroupedMatmul" in k) / 1000)
            lines.append(f"| {row['history']//1024}K/B{row['batch']} | {side} | " +
                         " | ".join(f"{v:.3f}" for v in totals) + " |")
    lines += [
        "", "专家GMM任务合计可能重叠，包含在FFN中，不得再相加。主图区间也不是完整_model_forward事件边界。",
        "本轮未采实际专家索引；不同event模式会影响profile，不把时长变化独立归因为路由或硬件效率。",
        "本轮模型只补128K/B16、8K/B40，不能把其结果拼进旧版七档表。",
    ]
    (ROOT / "MODEL_GAP.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
