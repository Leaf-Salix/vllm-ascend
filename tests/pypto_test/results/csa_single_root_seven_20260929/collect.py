"""复用已验证两档与Native，收齐同源码PTO七档，不伪称同次A/B。"""

import functools
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
CASES = ((131072, 4), (131072, 8), (131072, 16), (131072, 24), (8192, 16), (8192, 24), (8192, 32))
REUSED = {(131072, 16), (8192, 24)}
PAIR = ROOT.parent / "csa_score_single_root_20260929"
NATIVE = ROOT.parent / "csa_native_inplace_seven_20260929"


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finished(folder):
    task = (folder / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"等待原任务完成，不能拼接未完成结果：{task} {status}")
    return task, status


def main():
    task, status = finished(ROOT)
    pair_task, _ = finished(PAIR)
    native_task, _ = finished(NATIVE)
    source = read(ROOT / "source.json")
    original = read(NATIVE / "evidence.json")
    native_cases = {(c["history"], c["batch"]): c for c in original["cases"]}
    if set(native_cases) != set(CASES) or original["task"] != native_task:
        raise ValueError("Native基线不是已完成的当前七档")
    base = load("single_root_seven_base", NATIVE / "collect.py")
    base.ROOT = ROOT
    pair = load("single_root_seven_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    metrics = load("single_root_seven_metrics", ROOT.parent / "csa_score_segment_ub_20260929/collect.py")
    worker = load("single_root_seven_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("single_root_seven_helper", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    result = {"task": task, "task_status": status, "source": source,
              "reused_pair_task": pair_task, "native_task": native_task,
              "native_compile_call": original["native_compile_call"], "cases": [],
              "scope": "同源码PTO：两档复用、五档补测；Native复用已有最新标准，非同次A/B或模型forward"}
    for history, batch in CASES:
        reused = (history, batch) in REUSED
        folder = (PAIR if reused else ROOT) / f"h{history}_b{batch}"
        timing = folder / "timing/candidate"
        report = read(timing / "report.json")
        if (report["source"], report["variant"], report["history"], report["batch"], report["side"]) != (
                source["source"], source["variant"], history, batch, "pto"):
            raise ValueError("错用PTO源码、私有包或档位")
        package = "vllm_ascend.ops.pypto." + source["variant"].removeprefix("pkg:")
        if (report["implementation_package"] != package or not report["compiler"]["pto_dispatch_calls"]
                or report["implementation_source"] != str(Path(source["source"]) /
                                                          (package.replace(".", "/") + "/service.py"))):
            raise ValueError("实际PTO模块或dispatch不符")
        checks = report["timing"]["eager_comparison"]
        if set(checks) != base.STATES or any(v["status"] != "PASS" for v in checks.values()):
            raise ValueError("PTO图重放状态未通过")
        pto = pair.analyze_side(timing)
        if pto["topk"]["structural_errors"]:
            raise ValueError("Top-K结构错误")
        samples = report["timing"]["samples_us"]
        if len(samples) != 20:
            raise ValueError("正式采样数发生变化")
        pto.update(samples_us=samples, p95_over_p50=pto["us_p95"] / pto["us_p50"],
                   max_over_p50=pto["us_max"] / pto["us_p50"],
                   over_p50_5pct=sum(v > 1.05 * pto["us_p50"] for v in samples))
        trace = read(Path(pto["profile_json"]))
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        if not any(e.get("ph") == "X" for e in events):
            raise ValueError("缺少PyTorch trace")
        native_case = native_cases[(history, batch)]
        native_folder = NATIVE / f"h{history}_b{batch}"
        native = native_case["sides"]["native"]
        for key in ("cann", "requested"):
            if native[key] != pto[key]:
                raise ValueError(f"两侧环境或负载不同：{key}")
        native_profile = base.native_scope(read(native_folder / "native/report.json"),
                                           Path(native["profile_csv"]))
        native_incore = base.native_scope(read(native_folder / "native_incore/report.json"),
                                          Path(native_case["native_incore"]["profile_csv"]), super_kernel=False)
        dfx = read(folder / "swimlane/candidate/report.json")
        windows = dfx["swimlane_windows"]
        if (dfx["status"], dfx["variant"], dfx["history"], dfx["batch"], len(windows),
                dfx["effective_weight_nz_mode"], dfx["deterministic_level"],
                dfx["pto_reduction"]["atomic_add"]) != ("MEASURED", source["variant"], history, batch, 4, 2, 0, 0):
            raise ValueError("四窗DFX配置/覆盖不完整")
        if any(not w["exported"] or (w["layer_index"], w["compact_metadata_policy"],
                                    w["input_source"], w["execution"]) != (
                4, "reuse", "formal_layer_weights_synthetic_history", "graph_replay") for w in windows):
            raise ValueError("DFX捕获范围改变")
        coefficient_workers = 48 if history == 8192 and batch in (16, 32) else batch
        summaries = [metrics.schedule_window(Path(w["merged_swimlane"]), coefficient_workers, helper, worker)
                     for w in windows]
        for window in summaries:
            tasks = window["tasks"]
            if "merge_norm" in tasks or any(tasks[name]["blocks"] != blocks for name, blocks in (
                    ("qk_pv_aic", 24), ("qk_pv_aiv", 48), ("indexer_score_topk_native_pair_aic", 24),
                    ("indexer_score_topk_native_pair_aiv", 48), ("indexer_topk_query_merge", 48))):
                raise ValueError("任务结构或worker覆盖改变")
        means = {name: statistics.mean(w["tasks"][name]["kernel_mean_us"] for w in summaries) for name in base.TASKS}
        result["cases"].append({"history": history, "batch": batch, "sides": {"native": native, "pto": pto},
                                "native_profile": native_profile, "native_incore": native_incore,
                                "worker_windows": summaries, "incore_means": means,
                                "pto_task": pair_task if reused else task, "pto_reused": reused,
                                "pto_self_graph_state_checks": checks,
                                "same_device_as_native": pto["device"] == native["device"],
                                "csa_change_pct": 100 * (pto["mean_us"] / native["mean_us"] - 1)})
    groups = {str(h): statistics.mean(c["csa_change_pct"] for c in result["cases"] if c["history"] == h)
              for h in (131072, 8192)}
    result["history_changes_pct"] = groups
    result["weighted_8_2_csa_change_pct"] = .8 * groups["131072"] + .2 * groups["8192"]
    result["pto_self_graph_state_status"] = "PASS"
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    base.write_summary(result)
    base.write_report(result)
    base.write_task_details(result)
    provenance = (
        f"Native复用任务{native_task}的最新标准基线；PTO两档复用{pair_task}，五档补测{task}。\n"
        "PTO为同一冻结源码；各侧设备/来源在summary/evidence中逐项保留。\n"
        "这是当前已有结果对照，不是同次Native/PTO A/B，不用于归因单项优化收益。\n"
        "单根方案的严格同卡局部A/B见../csa_score_single_root_20260929/RESULTS.md。\n"
        "七档PTO各自图重放的八类状态通过；旧七档未保存完整state，不能声称七档跨版本状态一致。\n"
        "跨版本状态依据仅限已完成的两档A/B和尾段/padding；不是Native/PTO或模型token/DSpark验收。\n\n"
    )
    for name in ("RESULTS.md", "TASKS.md"):
        p = ROOT / name
        text = p.read_text()
        if name == "RESULTS.md":
            text = text.replace("# 新Native基线与Sparse融合后的七档", "# 单根Indexer采用后的PTO七档与现有Native")
        title, body = text.split("\n\n", 1)
        p.write_text(title + "\n\n" + provenance + body)
    compact = read(ROOT / "summary.json")
    for dst, src in zip(compact["cases"], result["cases"]):
        for key in ("pto_task", "pto_reused", "same_device_as_native", "pto_self_graph_state_checks"):
            dst[key] = src[key]
    (ROOT / "summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
