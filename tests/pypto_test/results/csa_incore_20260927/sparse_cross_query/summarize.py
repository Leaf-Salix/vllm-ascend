"""Collect existing cross-query pilot results; no device execution."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
CASES = ((131072, 4), (131072, 8), (131072, 16),
         (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load_module("csa_matrix_summary", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    collector = load_module("early_kv_summary", ROOT.parent / "sparse_kv_early/summarize.py")
    rows = collector.collect(common, "sparse_cross_query", CASES)
    checks = []
    for name, reference in (
        ("standalone", "固定Native Sparse输入；对Native输出的误差，非PTO基底逐bit检查"),
        ("mixed_tail", "B9均匀attention解析值；有效→全无效→有效query及不均分尾部"),
        ("zero_work_cores_b3", "B3均匀attention解析值；T18、6个AIC零工作量"),
    ):
        path = ROOT / name / "report.json"
        if path.exists():
            checks.append({"name": name, "reference": reference, "report": str(path),
                           "comparison": json.loads(path.read_text())["comparison"]})
    result = {
        "base_revision": "c160cabe",
        "operator_revision": "da2e2368",
        "patch": "candidate.patch",
        "scope": "同一da2e2368算子七档单卡实测；每档独立采集，不代表16卡整模型验收",
        "task_ids": ["task_20260927_165242_22114026823", "task_20260927_165609_22367535890",
                     "task_20260927_170134_22993926622"],
        "fixed_input_vs_baseline": {
            "status": "PASS",
            "elements": 7864320,
            "comparison": "torch.equal(actual, baseline)",
            "baseline": "../sparse_kv_early/gated_standalone/output.pt",
            "actual": "standalone/output.pt",
            "evidence": "task_20260927_165242_22114026823 stdout: CROSS_QUERY_FIXED_INPUT_AND_MIXED_TAIL_BIT_EQUAL_PASS",
        },
        "diagnostics": checks,
        "cases": rows,
    }
    (ROOT / "cases.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if len(rows) == len(CASES) and all(len(row["swimlane"]) == 4 for row in rows):
        write_matrix(rows)
    for row in rows:
        aic = [w["qk_pv_aic"]["mean_us"] for w in row["swimlane"]]
        merge = [w["merge_norm"]["mean_us"] for w in row["swimlane"]]
        print(row["history"], row["batch"], row["timing"]["body"],
              "AIC", (min(aic), max(aic)) if aic else "pending",
              "merge", (min(merge), max(merge)) if merge else "pending")


def write_matrix(rows):
    baseline = json.loads((RESULTS / "csa_split_optimization_20260927/indexer_progress_v10.json").read_text())
    old = {(r["history"], r["batch"]): r["timing"]["body"]["mean_us"] for r in baseline}
    lines = [
        "# 核内优化当前七档：da2e2368", "",
        "2026-09-27。七档均为同一套性能版算子，累计包含B40 KV投影、按工作量早发布KV及跨query流水。",
        "对V10的变化是累计变化，不是跨query单项收益；本页不覆盖V10基线。",
        "单卡正式layer4权重、合成输入/历史、S6/TP1/mode2/atomic1/确定性0、EPLB关，复用第二CSA层metadata。",
        "5次预热、20次无profiler设备计时；本体含HC_pre→norm→CSA→HC_post，完整PTO另含拆分/写回。", "",
        "| H / B | 同轮Native μs | 当前本体 μs | 本体对Native | V10本体 μs | 当前对V10 | 当前本体p95 μs | 完整PTO μs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        h, b = row["history"], row["batch"]
        t = row["timing"]
        current, native, previous = t["body"]["mean_us"], t["native"]["mean_us"], old[h, b]
        lines.append(f"| {h // 1024}K / {b} | {native:.2f} | {current:.2f} | {100*(current/native-1):+.2f}% | "
                     f"{previous:.2f} | {100*(current/previous-1):+.2f}% | {t['body']['p95_us']:.2f} | {t['pto_full']['mean_us']:.2f} |")
    lines += ["", "本体七档均低于同轮Native，但完整PTO七档仍慢；没有消除拆分/写回成本。",
              "128K/B4、B8及8K/B16未获得本体收益。128K/B16的长尾仍在，不能只取正常窗口。", "",
              "## 当前Sparse核内", "",
              "四个独立DFX窗口，各窗口对24个AIC/48个AIV block取均值，再报范围。不是p50/p95，三列不能相加。", "",
              "| H / B | qk_pv AIC μs | qk_pv AIV μs | merge_norm AIV μs |",
              "| --- | ---: | ---: | ---: |"]
    for row in rows:
        intervals = []
        for task in ("qk_pv_aic", "qk_pv_aiv", "merge_norm"):
            values = [w[task]["mean_us"] for w in row["swimlane"]]
            intervals.append(f"{min(values):.2f}–{max(values):.2f}")
        lines.append(f"| {row['history']//1024}K / {row['batch']} | " + " | ".join(intervals) + " |")
    lines += ["", "## 功能与数值", "",
              "| H / B | 保护区失败 | Top-K结构错误 | 非有限值 | 输出max_abs | 输出RMSE | Top-K集合替换 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        t = row["timing"]
        lines.append(f"| {row['history']//1024}K / {row['batch']} | {len(t['guard_failures'])} | "
                     f"{len(t['topk']['structural_errors'])} | {t['output']['nonfinite']} | "
                     f"{t['output']['max_abs']:.8f} | {t['output']['rmse']:.9f} | {t['topk']['replaced_indices']} |")
    lines += ["", "对Native零容差仍FAIL，Top-K集合替换不是metadata/保护区错误；未据此宣布精度验收通过。",
              "固定Native输入的Sparse对基底PTO逐bit一致，B9混合无效/尾部及B3零工作量核检查通过，见[实现记录](README.md)。",
              "未做本轮16卡逐token、DSpark及decode forward验收；750 μs最终目标尚未达到。", "",
              "## 证据", "",
              "[逐项计时、误差及全部28个泳道路径](cases.json)。三次任务均退出0，任务ID见该文件。",
              "[前三档及功能检查](README.md)、[剩余四档脚本](run_remaining.sh)、[离线汇总脚本](summarize.py)。",
              "Native总区间为每档同轮无profiler测量；历史Native分项trace仅用于原差距分析，未冒充本轮新采。", ""]
    (ROOT / "MATRIX.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
