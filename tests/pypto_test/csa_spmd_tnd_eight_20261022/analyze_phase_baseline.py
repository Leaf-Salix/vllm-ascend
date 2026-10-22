# SPDX-License-Identifier: Apache-2.0
"""Summarize current SPMD diagnostic phases without creating performance samples."""

import hashlib
import json
import statistics
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    report = dict(
        performance_eligible=False,
        baseline="baseline.json",
        limitations=[
            "Instrumentation perturbs Scalar/cache timing and scheduling",
            "Intervals start at the preceding published phase, which can differ across roles",
            "GM wait time excludes FFTS/FIFO waits; remaining time is not pure arithmetic",
            "Cross-window medians do not form one invocation timeline",
        ],
        cases={},
    )
    phases = ("SCORE_CUBE", "SCORE_VECTOR", "ATTN_CUBE", "ATTN_VECTOR", "OB")
    for key, name in (("C04", "h131072_b16_s6"), ("C08", "h131072_b16_mixed")):
        source = here / "runs" / (name + "_baseline_phase_v1") / "result.json"
        data = json.loads(source.read_text())
        assert data["passed"] and data["diagnostic_phase_clock"] and not data["performance_eligible"]
        assert "statistics_us" not in data and "ratio_to_frozen" not in data
        assert len(data["phase_windows"]) == 3
        for check in data["checks"].values():
            assert all(check["guards"].values()) and all(v["bit_equal"] for v in check["numerical"].values())
        historical = Path(baseline["cases"][key]["source_report"]).parent
        (cpp,) = historical.glob("build_output/decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
        instrumented_original = source.parent / "instrumentation/original.cpp"
        # Different compiler SSA identifiers are not silently normalized away.
        source_equality = instrumented_original.read_bytes() == cpp.read_bytes()

        def without_locations(text):
            return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("// pto:"))

        executable_equality = without_locations(instrumented_original.read_text()) == without_locations(cpp.read_text())
        assert executable_equality, "Unexpected preinstrumentation executable change"
        windows = []
        for window in data["phase_windows"]:
            rows = json.loads(Path(window["raw"]).read_text())
            summary = {}
            for phase in phases:
                selected = [r for r in rows if r["name"] == phase and "interval_us" in r]
                assert len(selected) == (48 if phase.endswith("VECTOR") else 24)
                summary[phase] = dict(
                    completion_spread_us=max(r["notify_end_us"] for r in selected)
                    - min(r["notify_end_us"] for r in selected),
                    median_interval_us=statistics.median(r["interval_us"] for r in selected),
                    median_gm_wait_us=statistics.median(r["wait_us"] for r in selected),
                    latest_cores=[r["core"] for r in sorted(selected, key=lambda r: -r["notify_end_us"])[:4]],
                )
            windows.append(dict(raw=window["raw"], phases=summary))
        report["cases"][key] = dict(
            source_report=str(source),
            device_id=data["device_id"],
            windows=windows,
            same_preinstrumentation_cpp_as_historical=source_equality,
            same_preinstrumentation_executable_cpp=executable_equality,
            comparison_rule="Remove only full-line // pto: location comments; keep all executable code",
            original_sha256=hashlib.sha256(instrumented_original.read_bytes()).hexdigest(),
            historical_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        )
        print(
            key,
            "Score cube completion spread median us",
            statistics.median(w["phases"]["SCORE_CUBE"]["completion_spread_us"] for w in windows),
            "same preinstrumentation cpp",
            source_equality,
        )
    (here / "baseline_phase_analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
