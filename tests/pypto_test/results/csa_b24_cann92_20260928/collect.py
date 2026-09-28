"""Validate the new private-package B24 case and collect existing profiles."""

import importlib.util
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANT = "pkg:dsv4_csa_b24_cann92_3b27c7fd"
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
          "indexer.0", "indexer.1", "indexer_state.0"}


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_pass(values, label):
    if not values or any(row["status"] != "PASS" for row in values.values()):
        raise ValueError(label)


def main():
    task = (ROOT / "task.txt").read_text().strip()
    if subprocess.check_output(["task-submit", "--status", task], text=True).strip() != "completed (exit=0)":
        raise ValueError("Device task is not successful and terminal")
    metrics = load("seven_metrics", ROOT.parent / "csa_cann92_incore_seven_20260928/collect.py")
    worker = load("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    report, dfx = read(ROOT / "timing/report.json"), read(ROOT / "swimlane/report.json")
    expected = {"history": 131072, "batch": 24, "seed": 1024, "layer_index": 4,
                "variant": VARIANT, "weight_nz_mode": 2, "effective_weight_nz_mode": 2,
                "deterministic_level": 0, "hccl_deterministic": False, "status": "MEASURED",
                "checkpoint": "/data/model/DeepSeek-V4-Flash-0731-w8a8",
                "pto_reduction": {"atomic_add": 0, "qr_split_k": 1, "kv_split_k": 1}}
    for source in (report, dfx):
        if any(source[key] != value for key, value in expected.items()):
            raise ValueError("Configuration differs from frozen B24 case")
        if source["accuracy_fixture"]["kind"] != "per_physical_row" or set(source["pto_self"]) != STATES:
            raise ValueError("Fixture or eight-state contract differs")
        require_pass(source["pto_self"], "PTO eager replay")
        for guard in (*source["native_guards"], *source["pto_guards"]):
            require_pass(guard, "Metadata or external guard")
        if source["topk_selection"]["structural_errors"]:
            raise ValueError("Top-K structure")
        if any(row.get("nonfinite", 0) for row in source["pto_native"].values()):
            raise ValueError("Nonfinite state")
        for binding in source["weight_storage_binding"].values():
            if (binding["native_format"], binding["pto_format"], binding["same_data_ptr"]) != (29, 29, True):
                raise ValueError("Native root NZ address was not borrowed")
    timing = report["timing"]
    if (timing["status"], timing["iters"], timing["warmup"], timing["compact_metadata_policy"]) != (
        "MEASURED", 20, 5, "reuse"
    ):
        raise ValueError("Timing contract")
    values = {}
    for side in ("native", "pto"):
        if len(set(timing[side]["start_timestamps_raw"])) != 20:
            raise ValueError("Timing events did not advance")
        require_pass(timing[side]["guards"], side + " timing guards")
        if timing[side]["topk_selection"]["structural_errors"]:
            raise ValueError(side + " timing Top-K")
        values[side] = metrics.stats(timing[side]["samples_us"])
    require_pass(timing["pto"]["eager_comparison"], "PTO graph versus eager")
    if report["graph"]["status"] != "PASS" or [r["input"] for r in report["graph"]["replays"]] != ["A", "B", "A"]:
        raise ValueError("A-B-A graph replay")
    windows = dfx["swimlane_windows"]
    if len(windows) != 4:
        raise ValueError("Expected four DFX windows")
    for w in windows:
        if not w["exported"] or (w["layer_index"], w["compact_metadata_policy"], w["input_source"], w["execution"]) != (
            4, "reuse", "formal_layer_weights_synthetic_history", "graph_replay"
        ):
            raise ValueError("DFX contract")
    profiles = {side: metrics.profile(ROOT / "timing", side) for side in ("native", "pto")}
    workers = [metrics.worker_window(Path(w["merged_swimlane"]), worker) for w in windows]
    change = (values["pto"]["mean_us"] / values["native"]["mean_us"] - 1) * 100
    result = {"task": task, "operator": "3b27c7fd", "config": expected,
              "prepare": read(ROOT / "prepare.json"), "functional_status": "PASS",
              "scope": "Single-card self-replay, metadata/guards and A-B-A; not model token/DSpark acceptance",
              "ring_request": {"heap_mb": [256, 128, 256, 32], "task_window": 4096},
              "timing": values, "change_pct": change, "device": timing["device"],
              "weight_storage_binding": report["weight_storage_binding"], "pto_self": report["pto_self"],
              "native_arithmetic_diagnostics": report["pto_native"], "profiles": profiles,
              "worker_windows": workers, "sources": [str(ROOT / "timing/report.json"),
                                                      str(ROOT / "swimlane/report.json")]}
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# CANN9.2：128K/B24首次当前源码单卡对照", "",
             "正式layer4权重/合成历史；mode2/atomic0/det0/S6，完整HC_pre→HC_post。",
             "每侧预热5次，20次无profiler图计时，单位μs。", "",
             "| 实现 | 均值 | P50 | P95 | 最大值 |", "| --- | ---: | ---: | ---: | ---: |"]
    for side, data in values.items():
        lines.append("| " + side + " | " + " | ".join(f"{data[k]:.3f}" for k in (
            "mean_us", "p50_us", "p95_us", "max_us")) + " |")
    lines += ["", f"PTO完整CSA耗时变化{change:+.3f}%；新档自重放、图状态及保护区通过，Native数值差异单列。",
              "不与旧七档拼成同轮结果；本轮未运行新的EP16。", "", "## 核内与包络", "",
              "四DFX窗口指标均值；含DMA/内部等待，非纯算术。", "",
              "| Task | 核内均值 | 最慢核 | 包络 | 启动分散 | 单核最多份数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in (*metrics.TASKS, "proj_a_mm", "proj_b_mm", "hc_post"):
        rows = [w["tasks"][name] for w in workers]
        cells = [f"{statistics.mean(r[k] for r in rows):.3f}" for k in (
            "kernel_mean_us", "kernel_max_us", "worker_envelope_us", "start_spread_us")]
        lines.append("| " + " | ".join([name, *cells, str(max(r["max_blocks_per_core"] for r in rows))]) + " |")
    download = ROOT / "download"
    download.mkdir(exist_ok=True)
    lines += ["", "## JSON", "", "PyTorch profile与四DFX分别采集，不是同一次调用。", ""]
    for name, source in [("01_Native_PyTorch", profiles["native"]["trace"]),
                         ("02_PTO_PyTorch", profiles["pto"]["trace"]),
                         ("03_PTO_Swimlane_SingleCSA_SyntheticHistory", workers[0]["path"])]:
        target = download / f"128K_B24_{name}.json"
        if not target.exists():
            os.link(source, target)
        lines.append(f"- [{target.name}]({target.relative_to(ROOT)})")
    for i, w in enumerate(workers):
        lines.append(f"- [PTO泳道窗口{i}]({Path(w['path']).relative_to(ROOT)})")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("PASS", task, f"CSA change={change:+.3f}%")


if __name__ == "__main__":
    main()
