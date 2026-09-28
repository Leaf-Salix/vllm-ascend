"""Check the uncovered dual-query path for retained coefficient changes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402

STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0", "indexer.0", "indexer.1", "indexer_state.0"}


def read(path):
    return json.loads(path.read_text())


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same task: {status}")
    source = read(ROOT / "source.json")
    spec = importlib.util.spec_from_file_location(
        "coefficient_dual_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(4)
    sides, states = {}, {}
    for side in ("baseline", "candidate"):
        folder = ROOT / "h8192_b32/timing" / side
        report = read(folder / "report.json")
        expected = source["source_prefix"] + side
        package = source["variant"].removeprefix("pkg:")
        if (report["source"], report["variant"], report["batch"], report["history"], report["side"]) != (
            expected,
            source["variant"],
            32,
            8192,
            "pto",
        ):
            raise ValueError("Unexpected source, variant or shape")
        if (report["implementation_package"], report["implementation_source"]) != (
            "vllm_ascend.ops.pypto." + package,
            str(Path(expected) / "vllm_ascend/ops/pypto" / package / "service.py"),
        ):
            raise ValueError("Loaded a different implementation")
        sides[side] = helper.analyze_side(folder)
        sides[side]["samples_us"] = report["timing"]["samples_us"]
        if not report["compiler"]["pto_dispatch_calls"] or report["timing"]["topk_selection"]["structural_errors"]:
            raise ValueError("Missing PTO dispatch or invalid Top-K structure")
        states[side] = torch.load(folder / "states.pt", map_location="cpu", weights_only=True)
        if set(states[side]) != STATES:
            raise ValueError("Missing full state coverage")
    for key in ("device", "cann", "requested"):
        if sides["baseline"][key] != sides["candidate"][key]:
            raise ValueError(f"The two sides differ: {key}")
    checks = {
        name: compare_tensor(states["candidate"][name], states["baseline"][name], 0, 0) for name in sorted(STATES)
    }
    del states
    passed = all(value["status"] == "PASS" for value in checks.values())
    result = {
        "task": task,
        "task_status": status,
        "source": source,
        "history": 8192,
        "batch": 32,
        "state_status": "PASS" if passed else "FAIL",
        "state_checks": checks,
        "sides": sides,
        "scope": "One missing dual-query state check; no DFX, Native or model acceptance",
    }
    a, b = (sides[side] for side in ("baseline", "candidate"))
    change = 100 * (b["mean_us"] / a["mean_us"] - 1)
    result["csa_change_pct"] = change
    (ROOT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 双query系数改动的剩余状态检查",
        "",
        f"8K/B32，c93ec723→4ffccb7b，八类完整状态零容差：{result['state_status']}。",
        "两侧CANN9.2/mode2/atomic0/det0，同卡5预热/20次真实编译图计时。",
        "",
        "| CSA基线→当前μs | 变化 | P95μs | maxμs |",
        "| ---: | ---: | ---: | ---: |",
        f"| {a['mean_us']:.3f}→{b['mean_us']:.3f} | {change:+.3f}% "
        f"| {a['us_p95']:.3f}→{b['us_p95']:.3f} | {a['us_max']:.3f}→{b['us_max']:.3f} |",
        "",
        "本轮只补双query缺口，没有新增DFX或Native比较，不把该点拼入旧七档性能表。",
        "精度版与整模型token/DSpark仍须各自验收。[完整状态与计时](summary.json)。",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not passed:
        raise SystemExit("Dual-query cross-version state check failed")


if __name__ == "__main__":
    main()
