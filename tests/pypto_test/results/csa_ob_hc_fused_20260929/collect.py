"""比较两级收尾与融合收尾的总核内工作量、包络、CSA和完整状态。"""

import functools
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fusion_common", ROOT.parent / "csa_sparse_first_pv_20260929/collect.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.torch.set_num_threads(4)
    read, load = common.read, common.load
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"等待同一任务终态：{status}")
    source = read(ROOT / "source.json")
    pair = load("fusion_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    worker = load("fusion_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    canonical = worker.canonical
    worker.canonical = lambda name: ("proj_b_act_hc_post" if re.fullmatch(
        r"proj_b_act_hc_post(?:_\d+)?", canonical(name)) else canonical(name))
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("fusion_join", ROOT.parents[4] / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    result = {"task": task, "task_status": status, "source": source, "cases": [],
              "scope": "同卡两档局部A/B；不是Native、精度版CSA或模型token/DSpark验收"}
    for history, batch in source["cases"]:
        folder = ROOT / f"h{history}_b{batch}"
        sides, states = {}, {}
        for side in ("baseline", "candidate"):
            timing = folder / "timing" / side
            report = read(timing / "report.json")
            if (report["source"], report["variant"], report["batch"], report["history"], report["side"]) != (
                    source["source_prefix"] + "-" + side, source["variant"], batch, history, "pto"):
                raise ValueError("私有源码、档位或入口不匹配")
            value = sides[side] = pair.analyze_side(timing)
            value["samples_us"] = report["timing"]["samples_us"]
            value["p95_over_p50"] = value["us_p95"] / value["us_p50"]
            value["over_p50_5pct"] = sum(x > 1.05 * value["us_p50"] for x in value["samples_us"])
            if value["topk"]["structural_errors"]:
                raise ValueError("Top-K结构错误")
            states[side] = common.torch.load(timing / "states.pt", map_location="cpu", weights_only=True)
            if set(states[side]) != common.STATES:
                raise ValueError("缺少完整状态")
            dfx = read(folder / "swimlane" / side / "report.json")
            if (dfx["status"], dfx["variant"], dfx["history"], dfx["batch"],
                    dfx["effective_weight_nz_mode"], dfx["deterministic_level"],
                    dfx["pto_reduction"]["atomic_add"]) != ("MEASURED", source["variant"], history, batch, 2, 0, 0):
                raise ValueError("DFX配置不匹配")
            windows = dfx["swimlane_windows"]
            if len(windows) != 4 or any(not w["exported"] or w["execution"] != "graph_replay" for w in windows):
                raise ValueError("四窗口采集不完整")
            value["windows"] = [tail_window(Path(w["merged_swimlane"]), batch, side, helper, worker)
                                for w in windows]
        for key in ("device", "cann", "requested"):
            if sides["baseline"][key] != sides["candidate"][key]:
                raise ValueError(f"两侧配置不同：{key}")
        checks = {k: common.compare_tensor(states["candidate"][k], states["baseline"][k], 0, 0)
                  for k in sorted(common.STATES)}
        del states
        a, b = (sides[s] for s in ("baseline", "candidate"))
        changes = {k: 100 * (statistics.mean(w[k] for w in b["windows"]) /
                            statistics.mean(w[k] for w in a["windows"]) - 1)
                   for k in ("tail_kernel_work_us", "tail_span_us")}
        result["cases"].append({"history": history, "batch": batch, "sides": sides, "state_checks": checks,
                                "csa_change_pct": 100 * (b["mean_us"] / a["mean_us"] - 1),
                                "tail_change_pct": changes})
    result["state_status"] = "PASS" if all(v["status"] == "PASS" for c in result["cases"]
                                          for v in c["state_checks"].values()) else "FAIL"
    result["weighted_8_2_csa_pct"] = sum(w * c["csa_change_pct"] for w, c in zip((.8, .2), result["cases"]))
    result["weighted_8_2_tail_pct"] = {k: sum(w * c["tail_change_pct"][k]
                                             for w, c in zip((.8, .2), result["cases"]))
                                     for k in ("tail_kernel_work_us", "tail_span_us")}
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "cases"}
    summary["cases"] = []
    for case in result["cases"]:
        item = {k: v for k, v in case.items() if k != "sides"}
        item["sides"] = {s: {k: v[k] for k in ("device", "cann", "mean_us", "us_p50", "us_p95", "us_max",
                                               "samples_us", "p95_over_p50", "over_p50_5pct", "guards", "profile_json")}
                         for s, v in case["sides"].items()}
        for s, v in case["sides"].items():
            item["sides"][s]["windows"] = [{k: x for k, x in w.items() if k != "all_tasks"} for w in v["windows"]]
        summary["cases"].append(item)
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# O-B反量化与HC_post融合：两档对照", "",
             f"八类跨版本状态零容差：{result['state_status']}。单位μs，5预热20次正式事件，DFX独立四窗。", "",
             "| 档位 | CSA基线→候选 | 变化 | P95 | 总核内工作量 | 收尾跨度 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        tail = ["→".join(f"{statistics.mean(w[k] for w in s['windows']):.3f}" for s in (a, b))
                for k in ("tail_kernel_work_us", "tail_span_us")]
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {a['mean_us']:.3f}→{b['mean_us']:.3f} "
                     f"| {case['csa_change_pct']:+.3f}% | {a['us_p95']:.3f}→{b['us_p95']:.3f} | "
                     + " | ".join(tail) + " |")
    lines += ["", f"长短8:2 CSA变化{result['weighted_8_2_csa_pct']:+.3f}%；总核内工作量变化"
              f"{result['weighted_8_2_tail_pct']['tail_kernel_work_us']:+.3f}%。",
              "总核内工作量为全部相关worker核时之和，基线合并proj_b_act与hc_post；不跨不同worker数比较单核均值。",
              "收尾跨度为首个相关kernel start到最后end；包含波次与间隙，不等于核时总和或纯调度开销。",
              "独立DFX不与正式CSA样本直接相减。Native、精度版和整模型验收均未在本轮覆盖。",
              "[完整样本与状态](summary.json)、[官方join和全部任务](evidence.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if result["state_status"] != "PASS":
        raise SystemExit(1)


def tail_window(path, batch, side, helper, worker):
    analysis = helper._build_analysis(path.parent, path.parent, 2)
    summary = worker.summarize(path)
    tasks = summary["tasks"]
    expected = ({"proj_b_act": ((batch * 6 + 31) // 32) * 8, "hc_post": (batch * 6 + 3) // 4}
                if side == "baseline" else {"proj_b_act_hc_post": ((batch * 6 + 15) // 16) * 8})
    for task, count in expected.items():
        if tasks[task]["blocks"] != count:
            raise ValueError(f"{task}的worker数不符")
    if side == "candidate" and any(k in tasks for k in ("proj_b_act", "hc_post")):
        raise ValueError("融合路径仍运行独立收尾任务")
    names = {t: worker.canonical(n) for t, n in analysis.graph.name.items()}
    starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
    finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
    deps = {}
    for name in expected:
        ids = [t for t, n in names.items() if n == name]
        if len(ids) != 1:
            raise ValueError(f"{name}应为一个逻辑任务")
        task = ids[0]
        producers = [p for p in analysis.preds[task] if not helper._is_alloc(p, analysis)]
        if any(p not in finishes for p in producers):
            raise ValueError("收尾存在没有物理时戳的前置，不能归因ready")
        dependencies = [names[p] for p in producers]
        if name in ("proj_b_act", "proj_b_act_hc_post") and dependencies.count("proj_b_mm") != 8:
            raise ValueError("融合缺少八组O-B前置")
        if name in ("hc_post", "proj_b_act_hc_post") and not {"split_pre_post", "comb_sinkhorn"} <= set(dependencies):
            raise ValueError("融合缺少HC门控前置")
        deps[name] = {"producers": dependencies,
                      "last_producer_finish_to_first_start_us": starts[task] - max(finishes[p] for p in producers)}
    selected = {n: tasks[n] for n in expected}
    return {"path": str(path), "joined_rows": len(analysis.rows), "tasks": selected, "dependencies": deps,
            "tail_kernel_work_us": sum(t["blocks"] * t["kernel_mean_us"] for t in selected.values()),
            "tail_span_us": max(t["last_end_us"] for t in selected.values()) -
            min(t["first_start_us"] for t in selected.values()),
            "worker_span_us": summary["worker_span_us"], "all_tasks": tasks}


if __name__ == "__main__":
    main()
