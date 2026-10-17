# SPDX-License-Identifier: Apache-2.0
"""只从完整通过的同进程三方完整L1窗口汇总候选，不使用插桩/跨轮性能。"""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    jobs = json.loads((HERE / "jobs.json").read_text())
    rows = []
    for job in jobs["jobs"]:
        output = Path(job["output"])
        full = json.loads((output / "result.json").read_text())
        matrix = json.loads((output / "source_matrix/result.json").read_text())
        assert full["passed"] and full["full_csa_validated"] and full["performance_measured"]
        assert matrix["passed"] and matrix["same_process"] and matrix["same_root_addresses"]
        assert len(full["launches"]) == 3 and all(launch["passed"] for launch in full["launches"])
        assert all(value["passed"] for value in matrix["protocol"].values())
        assert len(matrix["windows"]) == 15
        medians = {}
        samples = {}
        for side in ("dynamic_pto", "spmd_reference", "spmd_candidate"):
            windows = [window for window in matrix["windows"] if window["side"] == side]
            assert len(windows) == 5 and all(window["cores"] == 72 for window in windows)
            if side.startswith("spmd_"):
                assert all(window["task_count"] == 1 for window in windows)
            samples[side] = [window["incore_span_us"] for window in windows]
            medians[side] = statistics.median(samples[side])
            assert medians[side] == matrix["statistics_us"][side]["median_us"]
        baseline = medians["dynamic_pto"]
        reference = medians["spmd_reference"]
        candidate = medians["spmd_candidate"]
        reference_policy = matrix["policies"]["spmd_reference"]
        candidate_policy = matrix["policies"]["spmd_candidate"].copy()
        assert candidate_policy.pop("ATTENTION_TRANSITIVE_DEPENDENCIES") is True
        assert candidate_policy == reference_policy
        rows.append(
            dict(
                **job,
                passed=True,
                medians_us=medians,
                samples_us=samples,
                candidate_reduction_vs_reference_pct=(1 - candidate / reference) * 100,
                candidate_reduction_vs_task_pct=(1 - candidate / baseline) * 100,
                target_met=candidate <= baseline * 0.7,
                matrix_check_names=list(matrix["checks"]),
                source_policy_only_difference="ATTENTION_TRANSITIVE_DEPENDENCIES=True",
            )
        )
    result = dict(
        passed=True,
        scope="完整精度/图/保护/代际通过；性能候选不采用",
        selected=False,
        default_enabled=False,
        reason="两档改善方向不一致，幅度约0.4%以内，未证实稳定收益",
        source_frozen=True,
        numerical_order_changed=False,
        signal_layout_release_and_ring_unchanged=True,
        precision_reference="原动态分task PTO；不代表Native精度或整网token通过",
        performance_scope="每档同卡同进程、三方轮换顺序、每方五窗完整incore首尾",
        rows=rows,
    )
    (HERE / "result_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
