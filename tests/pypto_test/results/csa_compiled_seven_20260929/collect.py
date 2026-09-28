"""Collect the terminal compiled seven-case matrix and independent task profiles."""
import csv
import importlib.util
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 4), (131072, 8), (131072, 16), (131072, 24), (8192, 24), (8192, 32), (8192, 40))
TASKS = ("indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv",
         "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv", "merge_norm")
FIELDS = ("kernel_mean_us", "kernel_max_us", "worker_envelope_us", "start_spread_us")


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Matrix is not successfully terminal: {status}")
    source = read(ROOT / "source.json")
    paired = load("paired_compilation", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    metrics = load("seven_metrics", ROOT.parent / "csa_cann92_incore_seven_20260928/collect.py")
    worker = load("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    result = {"task": task, "source": source, "cases": [],
              "scope": "Compiled attention halves and independent DFX/PMU; not EP16 or token/DSpark acceptance"}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}"
        sides = {side: paired.analyze_side(folder / side) for side in ("native", "pto")}
        for field in ("device", "cann", "requested"):
            if sides["native"][field] != sides["pto"][field]:
                raise ValueError(f"Paired configuration differs: {history}/{batch}/{field}")
        samples = {}
        tails = {}
        for side in sides:
            report = read(folder / side / "report.json")
            if report["source"] != source["source"]:
                raise ValueError("Measurement did not use the frozen source")
            if report["variant"] == source["variant"]:
                sides[side]["variant_evidence"] = {"selector": report["variant"]}
            elif side == "pto" and report["variant"] == "performance":
                # The original runner stored selected_variant(), which returns
                # the category "performance" for every pkg selector. Preserve
                # raw reports and require the actual compiler source paths.
                package = Path(source["source"]) / "vllm_ascend/ops/pypto" / source["variant"].removeprefix("pkg:")
                log_path = folder / side / "run.log"
                log = log_path.read_text()
                required = [str(package / name) for name in ("decode_csa.py", "decode_indexer.py")]
                if any(path not in log for path in required):
                    raise ValueError("Legacy category lacks frozen package compilation evidence")
                sides[side]["variant_evidence"] = {
                    "selector": source["variant"], "legacy_report_category": report["variant"],
                    "compiler_source_paths": required, "compiler_log": str(log_path),
                }
            else:
                raise ValueError("Measurement package differs")
            if report["timing"]["topk_selection"]["structural_errors"]:
                raise ValueError("Invalid Top-K structure")
            samples[side] = report["timing"]["samples_us"]
            median = statistics.median(samples[side])
            tails[side] = {
                "p95_over_p50": sides[side]["us_p95"] / median,
                "max_over_p50": sides[side]["us_max"] / median,
                "over_p50_5pct": sum(v > median * 1.05 for v in samples[side]),
                "note": "Descriptive tail indicators for these 20 samples; no EP16 stability claim",
            }
        if not sides["pto"]["compiler"]["pto_dispatch_calls"]:
            raise ValueError("No actual PTO service call")
        windows_report = read(folder / "swimlane/report.json")
        if windows_report["variant"] != source["variant"]:
            raise ValueError("DFX package differs")
        windows = windows_report["swimlane_windows"]
        if len(windows) != 4 or any(not w["exported"] or w["execution"] != "graph_replay" for w in windows):
            raise ValueError("Expected four exported graph replay windows")
        if any((w["layer_index"], w["compact_metadata_policy"], w["input_source"]) !=
               (4, "reuse", "formal_layer_weights_synthetic_history") for w in windows):
            raise ValueError("DFX measurement scope differs")
        summaries = [metrics.worker_window(Path(w["merged_swimlane"]), worker) for w in windows]
        means = {name: {field: statistics.mean(w["tasks"][name][field] for w in summaries)
                        for field in FIELDS} for name in TASKS}
        with Path(sides["native"]["profile_csv"]).open() as stream:
            native_rows = list(csv.DictReader(stream))
        native = [{key: row[key] for key in ("Name", "Type", "Duration(us)", "aicore_time(us)", "aiv_time(us)")}
                  for row in native_rows]
        for name in ("VllmQuantLightningIndexer", "SparseAttnSharedkv"):
            if sum(row["Type"] == name for row in native) != 1:
                raise ValueError(f"Expected one Native {name}")
        n, p = (sides[s]["mean_us"] for s in ("native", "pto"))
        result["cases"].append({"history": history, "batch": batch, "sides": sides, "samples_us": samples,
                                "csa_change_pct": 100 * (p / n - 1), "tail_indicators": tails, "native_profile": native,
                                "worker_windows": summaries, "incore_means": means})
    if len({case["sides"]["native"]["device"] for case in result["cases"]}) != 1:
        raise ValueError("Matrix promised one allocated device")
    groups = {str(h): statistics.mean(c["csa_change_pct"] for c in result["cases"] if c["history"] == h)
              for h in (131072, 8192)}
    result["history_changes_pct"] = groups
    result["weighted_8_2_csa_change_pct"] = .8 * groups["131072"] + .2 * groups["8192"]
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 当前七档：Native/PTO真实编译半层与核内对照", "",
             "同卡、CANN9.2/mode2/det0；PTO atomic0。5预热/20次无profiler计时；单位μs。", "",
             "| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO max |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        n, p = (case["sides"][s] for s in ("native", "pto"))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {n['mean_us']:.3f} | {p['mean_us']:.3f} "
                     f"| {case['csa_change_pct']:+.3f}% | {n['us_p95']:.3f}/{p['us_p95']:.3f} "
                     f"| {n['us_max']:.3f}/{p['us_max']:.3f} |")
    lines += ["", f"长短8:2变化率：{result['weighted_8_2_csa_change_pct']:+.3f}%。", "",
              "以下为独立profile窗口；Native QLI含融合归并，PTO merge单列。",
              "PMU和DFX边界不同，不能据此计算纯算术加速比或与正式CSA相减归因调度。", "",
              "| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO merge | "
              "Native Sparse AIC/AIV | PTO QK/PV AIC/AIV | PTO merge_norm |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        native = {r["Type"]: r for r in case["native_profile"]}
        def pmu(name, native=native):
            return f"{float(native[name]['aicore_time(us)']):.3f}/{float(native[name]['aiv_time(us)']):.3f}"
        m = case["incore_means"]
        def kt(name, m=m):
            return m[name]["kernel_mean_us"]
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {pmu('VllmQuantLightningIndexer')} "
                     f"| {kt(TASKS[0]):.3f}/{kt(TASKS[1]):.3f} | {kt(TASKS[2]):.3f} "
                     f"| {pmu('SparseAttnSharedkv')} | {kt(TASKS[3]):.3f}/{kt(TASKS[4]):.3f} | {kt(TASKS[5]):.3f} |")
    lines += ["", "| 档位 | PTO P95/P50 | PTO max/P50 | PTO超过P50的105%的样本数 |",
              "| --- | ---: | ---: | ---: |"]
    for case in result["cases"]:
        tail = case["tail_indicators"]["pto"]
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {tail['p95_over_p50']:.4f} "
                     f"| {tail['max_over_p50']:.4f} | {tail['over_p50_5pct']}/20 |")
    lines += ["", "上述尾部比例仅描述本轮20次采样，不能据此关闭历史间歇拖尾或EP16稳定性问题。",
              "每档两侧PyTorch JSON、PTO四个泳道、全部样本及状态限制见[evidence.json](evidence.json)。",
              "不等同整模型forward或EP16/token/DSpark验收；拖尾样本全部保留。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
