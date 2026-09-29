"""核对主动停止前已完成的长档，不把短档缺口当作完整两档实验。"""

import functools
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=130)":
        raise ValueError(f"需要确认主动终止后的真实终态：{status}")
    common = load("ob_partial_common", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    pair = load("ob_partial_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    metrics = load("ob_partial_metrics", ROOT.parent / "csa_score_segment_ub_20260929/collect.py")
    worker = load("ob_partial_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("ob_partial_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    torch.set_num_threads(4)
    source = common.read(ROOT / "source.json")
    result = {"task": task, "task_status": status, "source": source, "history": 131072, "batch": 16,
              "decision": "REJECTED", "termination": "长档核内明确回退，主动停止剩余短档以节省占卡",
              "scope": "只核对已完成的长档；短档没有完整A/B，不计算8:2", "sides": {}}
    states = {}
    for side in ("baseline", "candidate"):
        folder = ROOT / "h131072_b16"
        timing = folder / "timing" / side
        report = common.read(timing / "report.json")
        if (report["source"], report["variant"], report["history"], report["batch"]) != (
                source["source_prefix"] + "-" + side, source["variant"], 131072, 16):
            raise ValueError("长档来源不匹配")
        metrics_side = pair.analyze_side(timing)
        if set(report["timing"]["eager_comparison"]) != common.STATES or metrics_side["topk"]["structural_errors"]:
            raise ValueError("状态覆盖或Top-K结构异常")
        dfx = common.read(folder / "swimlane" / side / "report.json")
        windows = dfx["swimlane_windows"]
        if (dfx["status"], dfx["history"], dfx["batch"], dfx["variant"], len(windows)) != (
                "MEASURED", 131072, 16, source["variant"], 4):
            raise ValueError("长档四窗来源不匹配")
        summaries = [metrics.schedule_window(Path(w["merged_swimlane"]), 16, helper, worker) for w in windows]
        if any(w["tasks"]["proj_b_mm"]["blocks"] != 64 for w in summaries):
            raise ValueError("O-B worker覆盖异常")
        kernels = [w["tasks"]["proj_b_mm"]["kernel_mean_us"] for w in summaries]
        result["sides"][side] = {k: metrics_side[k] for k in (
            "device", "cann", "requested", "mean_us", "us_p50", "us_p95", "us_max", "guards", "profile_json")}
        result["sides"][side].update(samples_us=report["timing"]["samples_us"], kernel_us=kernels,
                                     windows=[{k: w[k] for k in ("path", "joined_rows")} for w in summaries])
        states[side] = torch.load(timing / "states.pt", map_location="cpu", weights_only=True, mmap=True)
        if set(states[side]) != common.STATES:
            raise ValueError("完整状态不齐")
    a, b = (result["sides"][s] for s in ("baseline", "candidate"))
    if any(a[k] != b[k] for k in ("device", "cann", "requested")):
        raise ValueError("两侧配置不同")
    result["state_checks"] = {k: common.compare_tensor(states["candidate"][k], states["baseline"][k], 0, 0)
                              for k in sorted(common.STATES)}
    result["state_status"] = "PASS" if all(v["status"] == "PASS" for v in result["state_checks"].values()) else "FAIL"
    result["csa_change_pct"] = 100 * (b["mean_us"] / a["mean_us"] - 1)
    result["kernel_change_pct"] = 100 * (statistics.mean(b["kernel_us"]) / statistics.mean(a["kernel_us"]) - 1)
    (ROOT / "partial_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# O-B激活驻留第一版：长档核内回退，主动停止", "",
             f"原任务终态{status}，因长档已明确回退主动停止，不能描述为完整两档退出0。",
             f"已完成128K/B16八类完整状态零容差：{result['state_status']}；自身图/保护区、8个官方DFX核对窗通过。", "",
             "| 指标 | 基线 | 候选 |", "| --- | ---: | ---: |",
             f"| CSA均值μs | {a['mean_us']:.3f} | {b['mean_us']:.3f} |",
             f"| P95μs | {a['us_p95']:.3f} | {b['us_p95']:.3f} |",
             f"| O-B核时四窗均值μs | {statistics.mean(a['kernel_us']):.3f} | {statistics.mean(b['kernel_us']):.3f} |",
             f"| O-B各窗范围μs | {min(a['kernel_us']):.3f}–{max(a['kernel_us']):.3f} "
             f"| {min(b['kernel_us']):.3f}–{max(b['kernel_us']):.3f} |", "",
             f"CSA变化{result['csa_change_pct']:+.3f}%，O-B核时{result['kernel_change_pct']:+.3f}%；"
             "核内范围不重叠，没有核内收益，不保留。",
             "短档只完成candidate计时，缺完整A/B，不计算8:2，不扩大边界/七档/EP16。",
             "生成码除A的GM读取减少外还将L0 K128双缓冲变成K256共用缓冲；"
             "结果否定这一组合，不能单独证明A驻留策略无效。",
             "下一步仅CPU构造保持K128及Right交替缓冲的A驻留版本，预检通过后再决定真机筛选。",
             "[完整样本、状态和窗口来源](partial_summary.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
