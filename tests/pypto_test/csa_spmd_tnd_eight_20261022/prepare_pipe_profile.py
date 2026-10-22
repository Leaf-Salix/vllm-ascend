# SPDX-License-Identifier: Apache-2.0
"""Reuse the current fixture and strict checks while replacing L1 sampling with PMU."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/pipe_sdk_harness_v2"
    shutil.copytree(
        here / "runs/sdk_measurement_harness_v1", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(here / "profile_current.py", target / "profile_current.py")
    path = target / "run_pto.py"
    source = path.read_text().replace("import statistics\n", "")
    source = replace_once(
        source,
        "    comparison = parser.add_mutually_exclusive_group(required=True)\n"
        '    comparison.add_argument("--calibrate-baseline", action="store_true")\n'
        '    comparison.add_argument("--reference-result", type=Path)\n',
        "",
    )
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n    args.calibrate_baseline = False\n    args.reference_result = None\n",
    )
    source = replace_once(
        source,
        '        "scope": "TND-SPMD complete 72-role single-MIX L1; SDK-matched comparison",\n',
        '        "scope": "Current TND-SPMD aggregate hardware PipeUtilization; no L1 sampling",\n'
        '        "performance_eligible": False,\n        "diagnostic_hardware_profile": True,\n',
    )
    source = replace_once(
        source,
        '        "sampling": {"warmup": 3, "valid_windows": 10, "replays_per_window": 1},\n',
        '        "sampling": {"warmup": 3, "pmu_steps": 3, "replays_per_step": 1},\n',
    )
    source = replace_once(
        source,
        "            graphs = {}\n",
        '            report["kernel_configs"] = {side: str(path) for side, path in tables.items()}\n'
        "            graphs = {}\n",
    )
    start = source.index("                records = []\n")
    end = source.index('            save("complete")\n', start) + len('            save("complete")\n')
    source = (
        source[:start]
        + """                from profile_current import collect, summarize_csv

                save("hardware_pipe_sampling")
                report["hardware_profile"] = collect(
                    graph=graphs["candidate"], entry=entries["candidate"],
                    reset=lambda: restore(fixture), stream=stream, output=args.output,
                )
                for side, graph in graphs.items():
                    restore(fixture)
                    graph.replay()
                    stream.synchronize()
                    state, check = snapshot(side)
                    numerical = compare(state, expected)
                    report["checks"][side + "_post_pmu"] = {"guards": check, "numerical": numerical}
                    if not all(check.values()) or not all(v["status"] == "PASS" for v in numerical.values()):
                        raise AssertionError("PMU replay changed data or guards")
                report["hardware_profile"].update(summarize_csv(report["hardware_profile"]["kernel_details"]))
                report["hardware_profile"]["passed"] = True
            report["passed"] = True
            save("complete")
"""
        + source[end:]
    )
    for forbidden in ("pypto.torch.begin_dfx()", 'report["statistics_us"]', 'report["ratio_to_frozen"]'):
        assert forbidden not in source
    ast.parse(source)
    path.write_text(source)
    launcher = replace_once(
        (here / "launch_priority_phase.sh").read_text(),
        'harness="$here/runs/phase_sdk_harness_v1"',
        'harness="$here/runs/pipe_sdk_harness_v2"',
    )
    (here / "launch_pipe_profile_v2.sh").write_text(launcher)
    subprocess.run(["bash", "-n", str(here / "launch_pipe_profile_v2.sh")], check=True)
    report = dict(
        parent="sdk_measurement_harness_v1",
        performance_eligible=False,
        shared_toolchain=json.loads((target / "expected_sdk.json").read_text()),
        scope="Three current-implementation PMU graph replays with pre/post strict validation",
        unchanged=(
            "Original frozen correctness reference, complete fixture/weights, compile options, graph and guard checks"
        ),
        additions="Actual kernel-config identity, whole-72-role signal generation and padding checks, PMU raw rows",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        reference_api="https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html",
    )
    with (here / "pipe_sdk_harness_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
