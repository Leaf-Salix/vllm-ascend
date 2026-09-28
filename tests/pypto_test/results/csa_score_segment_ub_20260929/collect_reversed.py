"""Read the single reversed-order long-case timing check; no device work."""
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    task = (ROOT / "reversed_task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same task: {status}")
    spec = importlib.util.spec_from_file_location(
        "compiled_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sides = {}
    for side in ("baseline", "candidate"):
        folder = ROOT / "reversed_order/h131072_b16/timing" / side
        report = read(folder / "report.json")
        first = read(ROOT / "h131072_b16/timing" / side / "report.json")
        for field in ("source", "variant", "implementation_source", "implementation_package",
                      "batch", "history", "cann", "requested", "side"):
            if report[field] != first[field]:
                raise ValueError(f"Reversed check changed {side}/{field}")
        sides[side] = module.analyze_side(folder)
        sides[side]["samples_us"] = report["timing"]["samples_us"]
    a, b = (sides[side] for side in ("baseline", "candidate"))
    for field in ("device", "cann", "requested"):
        if a[field] != b[field]:
            raise ValueError(f"Reversed pair mismatch: {field}")
    change = 100 * (b["mean_us"] / a["mean_us"] - 1)
    result = {"task": task, "status": status, "order": ["candidate", "baseline"],
              "sides": sides, "csa_change_pct": change,
              "scope": "Same frozen operators; timing-only order check, no new DFX or cross-version states"}
    (ROOT / "reversed_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 长档反向顺序计时", "",
             "同一冻结算子、CANN9.2/mode2/atomic0/det0；先候选、后基线，各5预热/20次。",
             "本次未增加DFX或跨版本状态检查；首轮完整状态/泳道证据仍见RESULTS.md。", "",
             "| 档位 | CSA基线→候选μs | 变化 | P95μs | maxμs |",
             "| --- | ---: | ---: | ---: | ---: |",
             f"| 128K/B16 | {a['mean_us']:.3f}→{b['mean_us']:.3f} | {change:+.3f}% "
             f"| {a['us_p95']:.3f}→{b['us_p95']:.3f} | {a['us_max']:.3f}→{b['us_max']:.3f} |", "",
             "不把两轮不同采样交叉配对，不用本轮计时给首轮Score/merge核时做因果归因。",
             "[逐次计时和配置](reversed_summary.json)。"]
    (ROOT / "REVERSED.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
