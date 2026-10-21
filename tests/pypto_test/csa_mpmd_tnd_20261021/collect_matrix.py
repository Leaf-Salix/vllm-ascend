# SPDX-License-Identifier: Apache-2.0
"""收集固定八档；缺档/失败时不计算目标G，不混用不同额外控制组的加权收益。"""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    frozen = json.loads((here / "cases.json").read_text())
    plan = json.loads(args.plan.read_text())
    expected = {case["name"]: case for case in frozen["cases"]}
    assert len(expected) == 8 and set(plan["results"]) == set(expected)
    report = {
        "candidate": plan["candidate"],
        "reference_commit": plan["reference_commit"],
        "scope": "固定八档完整CSA L1、数值和保护区；不包括B1额外边界/连续metadata矩阵/16卡整模型",
        "toolchain": plan["toolchain"],
        "note": plan["note"],
        "status": "incomplete",
        "rows": [],
        "incomplete": [],
        "weighted_geomean": None,
        "performance_gate_met": False,
    }
    candidate_path = (here / "experiments" / plan["candidate"] / "operator").resolve()
    for name, case in expected.items():
        path = Path(plan["results"][name])
        if not path.exists():
            report["incomplete"].append({"case": name, "result": str(path), "reason": "result尚不存在"})
            continue
        result = json.loads(path.read_text())
        if result["stage"] != "complete" or not result["passed"]:
            report["incomplete"].append({"case": name, "result": str(path), "reason": result["stage"]})
            continue
        assert result["case"] == case, f"{name}: 场景发生变化"
        assert result["reference_commit"] == frozen["reference_commit"] == plan["reference_commit"]
        assert Path(result["sources"]["mpmd"]).resolve() == candidate_path
        assert result["scope"] == "full CSA L1" and not result["enable_dep_gen"]
        sides = ["spmd", "mpmd"]
        if "mpmd_control" in result["statistics_us"]:
            sides.append("mpmd_control")
        for side in sides:
            samples = result["statistics_us"][side]["samples"]
            windows = [window["incore_span_us"] for window in result["windows"] if window["side"] == side]
            assert len(samples) == 5 and windows == samples
            assert all(math.isfinite(value) and value > 0 for value in samples)
            for key, value in (("min", min(samples)), ("mean", statistics.mean(samples)), ("max", max(samples))):
                assert math.isclose(value, result["statistics_us"][side][key], rel_tol=1e-10, abs_tol=1e-8)
            for phase in ("eager", "graph", "post_l1"):
                check = result["checks"][f"{side}_{phase}"]
                assert check["guards"] and all(check["guards"].values())
                assert len(check["numerical"]) == 9 and all(v["status"] == "PASS" for v in check["numerical"].values())
        ratio = result["statistics_us"]["mpmd"]["mean"] / result["statistics_us"]["spmd"]["mean"]
        report["rows"].append(
            {
                "case": name,
                "history_label": case["history_label"],
                "batch": case["active_requests"],
                "result": str(path),
                "device_id": result["device_id"],
                "statistics_us": result["statistics_us"],
                "ratio": ratio,
                "extra_control_source": result["sources"]["mpmd_control"],
                "candidate_over_extra_control": result["candidate_over_control"],
                "all_bit_equal": all(
                    v["bit_equal"] for check in result["checks"].values() for v in check["numerical"].values()
                ),
            }
        )
    if not report["incomplete"]:
        long = [row["ratio"] for row in report["rows"] if row["history_label"] == 131072]
        short = [row["ratio"] for row in report["rows"] if row["history_label"] == 8192]
        assert len(long) == 6 and len(short) == 2
        long_log = statistics.mean(math.log(value) for value in long)
        short_log = statistics.mean(math.log(value) for value in short)
        overall = math.exp(0.8 * long_log + 0.2 * short_log)
        report["weighted_geomean"] = {
            "G128": math.exp(long_log),
            "G8": math.exp(short_log),
            "G": overall,
            "target": 0.9,
        }
        report["performance_gate_met"] = overall <= 0.9
        report["status"] = "complete"
        controls = {row["extra_control_source"] for row in report["rows"]}
        if len(controls) == 1 and all("mpmd_control" in row["statistics_us"] for row in report["rows"]):
            # 额外控制必须是同一个不可变包；每档的样本/数值/保护已在上面验证。
            # 旧矩阵混用v17/v23等不同控制时，禁止生成这一汇总。
            control_logs = {
                history: statistics.mean(
                    math.log(row["statistics_us"]["mpmd_control"]["mean"] / row["statistics_us"]["spmd"]["mean"])
                    for row in report["rows"]
                    if row["history_label"] == history
                )
                for history in (131072, 8192)
            }
            control_g = math.exp(0.8 * control_logs[131072] + 0.2 * control_logs[8192])
            report["same_round_control_weighted_geomean"] = {
                "G128": math.exp(control_logs[131072]),
                "G8": math.exp(control_logs[8192]),
                "G": control_g,
            }
            report["candidate_over_control_weighted_geomean"] = overall / control_g
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("| 场景 | SPMD min | SPMD mean | SPMD max | MPMD min | MPMD mean | MPMD max | MPMD/SPMD mean |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["rows"]:
        stats = row["statistics_us"]
        values = [stats[side][key] for side in ("spmd", "mpmd") for key in ("min", "mean", "max")]
        print(f"| {row['case']} | " + " | ".join(f"{value:.2f}" for value in values) + f" | {row['ratio']:.6f} |")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "incomplete", "weighted_geomean", "performance_gate_met")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
