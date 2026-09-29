"""比较首PV候选的完整状态、CSA/P95及四窗口真实核内耗时。"""

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
TITLE = "首PV块省略零累加项"
TARGETS = {"aic": "qk_pv_aic", "aiv": "qk_pv_aiv"}
DESCRIPTION = "只删除首块alpha乘零/加零分支，保留beta及sink作用，后续PV块、softmax、舍入和末块发布不变。"
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0", "indexer.0", "indexer.1", "indexer_state.0"}
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"等待同一任务成功完成：{status}")
    torch.set_num_threads(4)
    source = read(ROOT / "source.json")
    pair = load("first_pv_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    metrics = load("first_pv_metrics", ROOT.parent / "csa_score_segment_ub_20260929/collect.py")
    worker = load("first_pv_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("first_pv_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    evidence = {"task": task, "task_status": status, "source": source, "cases": []}
    for history, batch in source["cases"]:
        folder = ROOT / f"h{history}_b{batch}"
        sides = {}
        states = {}
        for name in ("baseline", "candidate"):
            timing_path = folder / "timing" / name
            report = read(timing_path / "report.json")
            if (report["source"], report["variant"], report["batch"], report["history"], report["side"]) != (
                    source["source_prefix"] + "-" + name, source["variant"], batch, history, "pto"):
                raise ValueError("错用私有源码或档位")
            if not report["backend_options"]["inplace_pass"]:
                raise ValueError("本轮必须开启inplace_pass")
            side = pair.analyze_side(timing_path)
            side["samples_us"] = report["timing"]["samples_us"]
            side["p95_over_p50"] = side["us_p95"] / side["us_p50"]
            side["over_p50_5pct"] = sum(v > 1.05 * side["us_p50"] for v in side["samples_us"])
            if side["topk"]["structural_errors"]:
                raise ValueError("Top-K结构错误")
            states[name] = torch.load(timing_path / "states.pt", map_location="cpu", weights_only=True)
            if set(states[name]) != STATES:
                raise ValueError("完整状态缺失")
            dfx = read(folder / "swimlane" / name / "report.json")
            if (dfx["status"], dfx["variant"], dfx["history"], dfx["batch"], dfx["effective_weight_nz_mode"],
                    dfx["deterministic_level"], dfx["pto_reduction"]["atomic_add"]) != (
                    "MEASURED", source["variant"], history, batch, 2, 0, 0):
                raise ValueError("DFX的配置或来源不符")
            windows = dfx["swimlane_windows"]
            if len(windows) != 4 or any(not w["exported"] or w["execution"] != "graph_replay" for w in windows):
                raise ValueError("缺少独立四窗图重放")
            side["windows"] = [metrics.schedule_window(Path(w["merged_swimlane"]), batch, helper, worker)
                               for w in windows]
            side["kernel_us"] = {}
            for core, task_name in TARGETS.items():
                blocks = {"qk_pv_aic": 24, "qk_pv_aiv": 48, "hc_post": (batch * 6 + 3) // 4,
                          "split_pre_post": min(16, (batch * 6 + 7) // 8),
                          "comb_sinkhorn": (batch * 6 + 7) // 8,
                          "mix_x_rms_norm": (batch * 6 + 7) // 8,
                          "indexer_score_topk_native_pair_aic": 24,
                          "indexer_score_topk_native_pair_aiv": 48,
                          "indexer_topk_query_merge": 48}[task_name]
                tasks = [w["tasks"][task_name] for w in side["windows"]]
                if any(t["blocks"] != blocks for t in tasks):
                    raise ValueError(f"{task_name}核心覆盖改变")
                side["kernel_us"][core] = [t["kernel_mean_us"] for t in tasks]
            if any("merge_norm" in w["tasks"] for w in side["windows"]):
                raise ValueError("两侧均应包含已保留的末块融合")
            sides[name] = side
        for field in ("device", "cann", "requested"):
            if sides["baseline"][field] != sides["candidate"][field]:
                raise ValueError(f"两侧配置不同：{field}")
        checks = {k: compare_tensor(states["candidate"][k], states["baseline"][k], 0, 0) for k in sorted(STATES)}
        del states
        a, b = (sides[s] for s in ("baseline", "candidate"))
        evidence["cases"].append({"history": history, "batch": batch, "sides": sides, "state_checks": checks,
                                  "csa_change_pct": 100 * (b["mean_us"] / a["mean_us"] - 1),
                                  "kernel_change_pct": {
                                      core: 100 * (statistics.mean(b["kernel_us"][core]) /
                                                   statistics.mean(a["kernel_us"][core]) - 1)
                                      for core in TARGETS}})
    evidence["state_status"] = "PASS" if all(v["status"] == "PASS" for c in evidence["cases"]
                                            for v in c["state_checks"].values()) else "FAIL"
    evidence["weighted_8_2_csa_pct"] = sum(w * c["csa_change_pct"] for w, c in zip((.8, .2), evidence["cases"]))
    evidence["weighted_8_2_kernel_pct"] = {
        core: sum(w * c["kernel_change_pct"][core] for w, c in zip((.8, .2), evidence["cases"]))
        for core in TARGETS}
    (ROOT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    summary = {k: v for k, v in evidence.items() if k != "cases"}
    summary["cases"] = []
    for case in evidence["cases"]:
        item = {k: v for k, v in case.items() if k != "sides"}
        item["sides"] = {s: {k: v[k] for k in ("mean_us", "us_p50", "us_p95", "us_max", "samples_us", "kernel_us",
                                               "p95_over_p50", "over_p50_5pct", "guards", "profile_json")}
                         for s, v in case["sides"].items()}
        summary["cases"].append(item)
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# " + TITLE, "", f"完整八类状态零容差：{evidence['state_status']}。", "",
             "同卡CANN9.2/mode2/atomic0/det0，inplace_pass=True；5预热20次正式事件，单位μs。", "",
             "| 档位 | CSA基线→候选 | 变化 | P95 | max | " + " | ".join(TARGETS.values()) + " |",
             "| --- | ---: | ---: | ---: | ---: |" + " ---: |" * len(TARGETS)]
    for case in evidence["cases"]:
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        core = [f"{statistics.mean(a['kernel_us'][c]):.3f}→{statistics.mean(b['kernel_us'][c]):.3f}"
                for c in TARGETS]
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {a['mean_us']:.3f}→{b['mean_us']:.3f} "
                     f"| {case['csa_change_pct']:+.3f}% | {a['us_p95']:.3f}→{b['us_p95']:.3f} "
                     f"| {a['us_max']:.3f}→{b['us_max']:.3f} | " + " | ".join(core) + " |")
    lines += ["", f"长短8:2 CSA变化：{evidence['weighted_8_2_csa_pct']:+.3f}%；核内变化："
              + "、".join(f"{TARGETS[k]} {v:+.3f}%" for k, v in evidence["weighted_8_2_kernel_pct"].items()) + "。",
              "", "核时包含核内流水等待，独立DFX不能与正式CSA样本直接相减。",
              DESCRIPTION,
              "采用还需依据四窗口分布判断真实核内收益，状态失败则禁止采用；不是Native或模型token/DSpark验收。",
              "[四窗核时及原样本](summary.json)、[原始解析与完整状态](evidence.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if evidence["state_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
