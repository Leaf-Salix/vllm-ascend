# SPDX-License-Identifier: Apache-2.0
"""单规格 30% 性能门槛；不把单规格通过等同于完整目标完成。"""

import math
import statistics


MAX_LATENCY_RATIO = 0.70
MIN_WINDOWS = 3


def evaluate(report):
    samples = {side: [row["incore_span_us"] for row in report.get("l1_windows", [])
                      if row["side"] == side] for side in ("dynamic_pto", "single_spmd")}
    measured = all(len(values) >= MIN_WINDOWS and
                   all(math.isfinite(value) and value > 0 for value in values)
                   for values in samples.values())
    eligible = bool(report.get("eligible_for_performance_target", False))
    correct = bool(report.get("passed") and report.get("graph_replay_validated")
                   and report.get("progress_passed") and report.get("workspace_guards_passed"))
    result = dict(max_latency_ratio=MAX_LATENCY_RATIO, minimum_windows=MIN_WINDOWS,
                  sufficient_samples=measured, uninstrumented_eligible=eligible,
                  completed_correctness_checks=correct, scope="仅本规格的同轮性能门槛",
                  passed=False)
    if measured:
        baseline, candidate = (statistics.median(samples[side])
                               for side in ("dynamic_pto", "single_spmd"))
        result.update(dynamic_pto_us=baseline, single_spmd_us=candidate,
                      target_us=baseline * MAX_LATENCY_RATIO,
                      latency_ratio=candidate / baseline,
                      latency_reduction_pct=(1 - candidate / baseline) * 100,
                      remaining_us=max(0, candidate - baseline * MAX_LATENCY_RATIO),
                      passed=eligible and correct and candidate <= baseline * MAX_LATENCY_RATIO)
    return result
