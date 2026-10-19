# SPDX-License-Identifier: Apache-2.0
"""汇总当前完整CSA的全72角色阶段时钟；不运行算子，不把插桩当性能验收。"""
import ast
import json
import statistics
from collections import defaultdict
from pathlib import Path


def phase_names(source):
    values = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("PHASE_"):
                    values[target.id] = node.value.value
    names = {v: k for k, v in values.items() if isinstance(v, int) and k != "PHASE_COUNT"}
    for prefix in ("OA", "O_QUANT"):
        base = values[f"PHASE_{prefix}_GROUP_BASE"]
        for group in range(8):
            names[base + group] = f"PHASE_{prefix}_GROUP_{group}"
    return names


def distribution(values):
    if not values:
        return None
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def main():
    here = Path(__file__).resolve().parent
    names = phase_names(here / "source_reference/runtime_abi.py")
    reports = []
    for directory in sorted(here.glob("device_*")):
        result_file = directory / "result.json"
        raw = directory / "phase_clock/phase_clock.json"
        if not result_file.exists() or not raw.exists():
            continue
        result = json.loads(result_file.read_text())
        if not result.get("passed"):
            continue
        rows = json.loads(raw.read_text())
        observed = sorted({row["core"] for row in rows})
        if observed != list(range(72)):
            raise ValueError(f"{directory.name}: 没有完整覆盖24 AIC和48 AIV")
        buckets = defaultdict(list)
        trace = []
        for row in rows:
            if "interval_us" not in row:
                continue
            role = "AIC" if row["core"] < 24 else "AIV"
            name = names.get(row["phase"], f"PHASE_{row['phase']}")
            buckets[(row["phase"], role)].append(row)
            trace.append(dict(
                name=name, ph="X", pid=0, tid=row["core"],
                ts=row["notify_start_us"] - row["interval_us"], dur=row["interval_us"],
                args=dict(wait_us=row["wait_us"], other_us=row["other_us"],
                          includes_probe_cost=True, pure_compute=False),
            ))
        summaries = []
        for (phase, role), group in sorted(buckets.items()):
            summaries.append(dict(
                phase=phase, name=names.get(phase, f"PHASE_{phase}"), role=role, roles=len(group),
                interval_us=distribution([r["interval_us"] for r in group]),
                stage_gm_wait_us=distribution([r["wait_us"] for r in group]),
                other_us=distribution([r["other_us"] for r in group]),
                first_interval_start_us=min(r["notify_start_us"] - r["interval_us"] for r in group),
                last_notify_start_us=max(r["notify_start_us"] for r in group),
                cores=[r["core"] for r in group],
            ))
        report = dict(
            case=directory.name, result=str(result_file), raw=str(raw),
            observed_cores=observed, source=str(here / "source_reference"),
            eligible_for_performance=False, includes_probe_cost=True,
            note="按相邻阶段发布点划分；other含FFTS/DMA/Scalar/fence，不是纯计算。含无业务发布者，不据中位数推算真实工作核数。",
            stages=summaries,
        )
        (directory / "phase_clock/stage_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        (directory / "phase_clock/stages_all72.json").write_text(
            json.dumps(dict(traceEvents=trace), ensure_ascii=False) + "\n"
        )
        reports.append(report)
        print(directory.name, "roles", len(observed), "stage_role_groups", len(summaries))
        for stage in summaries:
            if stage["other_us"]["max"] >= 25:
                print(stage["name"], stage["role"], "roles", stage["roles"],
                      "interval", stage["interval_us"], "wait", stage["stage_gm_wait_us"],
                      "other", stage["other_us"])
    (here / "STAGE_SUMMARY.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
