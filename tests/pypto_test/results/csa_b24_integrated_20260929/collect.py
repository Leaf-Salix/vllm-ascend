"""Collect B24's compiled A/B, Native reference and independent incore traces."""
import csv
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402

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
        raise RuntimeError(status)
    timing_task = (ROOT / "task_timing.txt").read_text().strip()
    timing_status = subprocess.check_output(["task-submit", "--status", timing_task], text=True).strip()
    if timing_status not in ("completed (exit=0)", "completed (exit=2)"):
        raise RuntimeError(timing_status)
    torch.set_num_threads(4)
    folder = ROOT / "final/h131072_b24"
    pair = load("compiled_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    metrics = load("seven_metrics", ROOT.parent / "csa_cann92_incore_seven_20260928/collect.py")
    worker = load("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    sides = {side: pair.analyze_side(folder / side)
             for side in ("pto_baseline", "native_candidate", "pto_candidate")}
    for key in ("device", "cann", "requested"):
        values = [side[key] for side in sides.values()]
        if any(v != values[0] for v in values[1:]):
            raise ValueError(f"Compiled configuration differs: {key}")
    states = {side: torch.load(folder / f"pto_{side}/states.pt", map_location="cpu", weights_only=True)
              for side in ("baseline", "candidate")}
    expected = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0", "indexer.0", "indexer.1", "indexer_state.0"}
    if any(set(value) != expected for value in states.values()):
        raise ValueError("Incomplete paired state coverage")
    checks = {name: compare_tensor(states["candidate"][name], value, 0, 0)
              for name, value in states["baseline"].items()}
    if any(check["status"] != "PASS" for check in checks.values()):
        raise ValueError(f"Cross-version state differs: {checks}")
    incore = {}
    for side in ("baseline", "candidate"):
        report = read(folder / "swimlane" / side / "report.json")
        windows = report["swimlane_windows"]
        if len(windows) != 4:
            raise ValueError("Expected four DFX windows")
        for window in windows:
            if not window["exported"] or window["execution"] != "graph_replay":
                raise ValueError("Invalid DFX capture")
        summaries = [metrics.worker_window(Path(w["merged_swimlane"]), worker) for w in windows]
        incore[side] = {"windows": summaries,
                        "means": {name: {field: statistics.mean(w["tasks"][name][field] for w in summaries)
                                         for field in FIELDS} for name in TASKS}}
    graph_report = read(folder / "graph/candidate/report.json")
    if graph_report["graph"]["status"] != "PASS":
        raise ValueError("B24 A-B-A failed")
    native_csv = Path(sides["native_candidate"]["profile_csv"])
    with native_csv.open() as stream:
        native = [r for r in csv.DictReader(stream) if r["Type"] in (
            "VllmQuantLightningIndexer", "SparseAttnSharedkv", "TransposeBatchMatMul")]
    native_reference = [{key: r[key] for key in ("Type", "Duration(us)", "aicore_time(us)", "aiv_time(us)")}
                        for r in native]
    before, current, n = (sides[s]["mean_us"] for s in ("pto_baseline", "pto_candidate", "native_candidate"))
    result = {"task": task, "timing_task": timing_task, "timing_task_status": timing_status,
              "history": 131072, "batch": 24, "baseline": "3b27c7fd",
              "candidate": "c93ec723", "status": "PASS", "sides": sides, "state_checks": checks,
              "incore": incore, "native_profile_reference": native_reference,
              "pto_ab_change_pct": 100 * (current / before - 1),
              "current_vs_native_pct": 100 * (current / n - 1),
              "limits": "Compiled single attention half, not EP16/token/DSpark acceptance; "
                        "Native PMU and PTO DFX scopes differ, not a pure arithmetic speed ratio."}
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 128K/B24：近期组合及真实编译 Native 对照", "",
             "同卡、CANN9.2/mode2/det0、PTO atomic0；5预热/20次无profiler图事件。单位μs。", "",
             "| 实现 | CSA均值 | P95 | 最大值 |", "| --- | ---: | ---: | ---: |"]
    for side, value in sides.items():
        lines.append(f"| {side} | {value['mean_us']:.3f} | {value['us_p95']:.3f} | {value['us_max']:.3f} |")
    lines += ["", f"PTO组合相对3b27c7fd：{result['pto_ab_change_pct']:+.3f}%；"
              f"当前PTO相对同配置Native：{result['current_vs_native_pct']:+.3f}%。", "",
              "独立四窗口DFX，核时包含内部等待；不与正式CSA直接相减归因调度。", "",
              "| Task | 核内均值 | 最慢核 | 包络 | 启动分散 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in TASKS:
        a, b = (incore[side]["means"][name] for side in ("baseline", "candidate"))
        lines.append("| " + " | ".join([name, *(f"{a[f]:.3f}→{b[f]:.3f}" for f in FIELDS)]) + " |")
    lines += ["", "跨版本八类PTO状态精确相同，当前入口图重放及A→B→A、metadata/保护区通过。",
              "Native det0数值差异单列，未完成两侧逐元素/整网token/DSpark验收。",
              "[Native热点PMU参考、完整证据与profile/泳道路径](evidence.json)"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
