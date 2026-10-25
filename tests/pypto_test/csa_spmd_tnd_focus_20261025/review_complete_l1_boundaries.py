# SPDX-License-Identifier: Apache-2.0
"""纯内存复现L1窗口误收；默认审查冻结v2，可指定修复版并要求拒绝反例。"""

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path


def baseline(layout):
    rows = [[i, 11, 1 if i < 24 else 2, 1000 + i, 2000 + i, 0] for i in range(72)]
    if layout == "separate":
        rows += [[i, 10, 1, 100 + i, 200 + i, 0] for i in range(24, 72)]
    return {
        "metadata": {
            "clock_freq_hz": 50_000_000,
            "num_cores": 72,
            "core_types": ["aic"] * 24 + ["aiv"] * 48,
            "run_boundaries": [[0, 90]],
            "dropped_run_boundaries": 0,
        },
        "chip_swimlane_level": 1,
        "aicore_tasks": rows,
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=here / "frozen_v2/complete_l1_v2.py")
    parser.add_argument("--output", type=Path, default=here / "independent_l1_boundary_review.json")
    parser.add_argument("--require-rejection", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module.parent))
    spec = importlib.util.spec_from_file_location("reviewed_complete_l1", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for layout in ("fused", "separate"):
        for mutation in (
            "valid",
            "boundary_empty",
            "boundary_future",
            "two_boundaries",
            "dropped_boundary",
            "nan_end",
            "inf_end",
            "nan_frequency",
            "wrong_init_roles",
            "different_dispatch_tokens",
        ):
            if mutation == "wrong_init_roles" and layout == "fused":
                continue
            raw = copy.deepcopy(baseline(layout))
            if mutation == "boundary_empty":
                raw["metadata"]["run_boundaries"] = [[]]
            elif mutation == "boundary_future":
                raw["metadata"]["run_boundaries"] = [[0, 3000]]
            elif mutation == "two_boundaries":
                raw["metadata"]["run_boundaries"].append([1, 3000])
            elif mutation == "dropped_boundary":
                raw["metadata"]["dropped_run_boundaries"] = 1
            elif mutation == "nan_end":
                raw["aicore_tasks"][0][4] = float("nan")
            elif mutation == "inf_end":
                raw["aicore_tasks"][0][4] = float("inf")
            elif mutation == "nan_frequency":
                raw["metadata"]["clock_freq_hz"] = float("nan")
            elif mutation == "wrong_init_roles":
                for i, row in enumerate(raw["aicore_tasks"][72:]):
                    row[0] = i  # 48个不同核，但错误地包含24个AIC。
            elif mutation == "different_dispatch_tokens":
                raw["aicore_tasks"][0][2] = 99  # 合法能力边界：不是kernel func_id。
            must_reject = mutation not in ("valid", "different_dispatch_tokens")
            try:
                result = module.summarize(raw, init_layout=layout)
                span = result["incore_span_us"]
                observed = "accepted"
                detail = span if math.isfinite(span) else str(span)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                observed = "rejected"
                detail = str(error)
            cases.append(
                dict(layout=layout, mutation=mutation, must_reject=must_reject, observed=observed, detail=detail)
            )
    report = {
        "module_reviewed": str(args.module),
        "device_execution": False,
        "files_modified_by_review": [str(args.output)],
        "cases": cases,
        "schema_source": "Simpler src/common/platform/shared/host/chip_swimlane_collector.cpp:1250",
        "run_boundary_schema": "[epoch, start_cycles]；单个boundary必须非空且start_cycles不晚于最早incore记录。",
        "dispatch_token_semantics": {
            "row_column_2": "per-core monotonic dispatch token，用于和AICPU scheduler记录join，不是func_id。",
            "source": "Simpler src/common/platform/include/common/chip_swimlane_profiling.h:102-109,156-161",
            "function_identity": (
                "L1原始记录不携带func_id；需deps.json kernel_ids[]和kernel_config后处理，当前采集没有deps。"
            ),
            "required_action": "不能按reg_task_id等于kernel func_id或所有核reg_task_id相同添加拒收规则。",
        },
        "retained_measurement_scope": (
            "仍从最早incore start到最晚incore end；boundary只验证窗口归属，不把AICPU启动边界纳入耗时。"
        ),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    bad = [r for r in cases if r["must_reject"] and r["observed"] != "rejected"]
    print(json.dumps(dict(tested=len(cases), invalid_accepted=len(bad), report=str(args.output))))
    if args.require_rejection:
        assert not bad, bad
        assert all(r["observed"] == "accepted" for r in cases if not r["must_reject"])


if __name__ == "__main__":
    main()
