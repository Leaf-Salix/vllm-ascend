"""Summarize the affected O projection and validate Native NZ address borrowing."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ("proj_a_mm", "quant", "proj_b_mm", "proj_b_act", "hc_post")
FIELDS = ("kernel_mean_us", "kernel_max_us", "worker_envelope_us", "start_spread_us")


def read(path):
    return json.loads(path.read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-short-batch", type=int)
    args = parser.parse_args()
    metrics = load_module("seven_metrics", ROOT.parent / "csa_cann92_incore_seven_20260928/collect.py")
    worker = load_module("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    cases = []
    lines = ["# WO_A Native NZ：受影响任务的核内对照", "",
             "四个独立DFX窗口的指标均值，单位μs。核时包含内部等待；包络含必要多波执行。",
             "正式CSA/P95另取20次无profiler样本，不与DFX相加。", "",
             "| 档位 | Task | 核内均值 | 最慢核 | 包络 | 启动分散 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    specs = [(131072, 16), (8192, 16)]
    if args.extra_short_batch is not None:
        specs.append((8192, args.extra_short_batch))
    for history, batch in specs:
        folder = ROOT / f"h{history}_b{batch}"
        summary = read(folder / "summary.json")
        if summary["status"] != "PASS":
            raise ValueError("State comparison has not passed")
        case = {"history": history, "batch": batch, "sides": {}}
        for side, data in summary["measurements"].items():
            report = read(folder / side / "report.json")
            binding = report["weight_storage_binding"]["wo_a"]
            expected = (29, 29, True, "NZ") if side == "candidate" else (29, 2, False, "ND")
            actual = tuple(binding[k] for k in ("native_format", "pto_format", "same_data_ptr", "root_layout"))
            if actual != expected:
                raise ValueError(f"{history}/{side} WO_A binding: {actual} != {expected}")
            windows = [metrics.worker_window(Path(w["path"]), worker) for w in data["worker_windows"]]
            case["sides"][side] = {
                "binding": binding, "timing": data["timing"],
                "windows": [{"path": w["path"], "tasks": {n: w["tasks"][n] for n in TASKS}}
                            for w in windows],
                "means": {n: {f: statistics.mean(w["tasks"][n][f] for w in windows) for f in FIELDS}
                          for n in TASKS},
            }
        case["o_a_kernel_change_pct"] = (
            case["sides"]["candidate"]["means"]["proj_a_mm"]["kernel_mean_us"]
            / case["sides"]["baseline"]["means"]["proj_a_mm"]["kernel_mean_us"] - 1
        ) * 100
        for name in TASKS:
            cells = [f"{case['sides']['baseline']['means'][name][f]:.3f}→"
                     f"{case['sides']['candidate']['means'][name][f]:.3f}" for f in FIELDS]
            lines.append("| " + " | ".join([f"{history // 1024}K/B{batch}", name, *cells]) + " |")
        cases.append(case)
    result = {"cases": cases, "weighted_o_a_kernel_change_pct":
              .7 * cases[0]["o_a_kernel_change_pct"] + .3 * cases[1]["o_a_kernel_change_pct"]}
    (ROOT / "o_projection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "O_PROJECTION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
