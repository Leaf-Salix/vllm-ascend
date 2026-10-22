# SPDX-License-Identifier: Apache-2.0
"""Freeze a diagnostic-only harness for the current eight-case SPMD baseline."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/measurement_harness"
    target = here / "runs/phase_baseline_harness"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    old = here.parent / "csa_spmd_tnd_opt_20261022"
    diagnostics = here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/compile_diagnostics.py"
    for source in (old / "phase_clock.py", diagnostics, here / "phase_probe_hooks.py"):
        shutil.copy2(source, target / source.name)
    source = (target / "run_pto.py").read_text()
    source = replace_once(
        source, "import types\n", "import types\n\nfrom phase_probe_hooks import compilation_probe, export_entry\n"
    )
    source = replace_once(
        source,
        '    frozen_baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())\n',
        "    import pypto\n"
        "    pypto.configure_cache(pypto.CacheConfig(enabled=False))\n"
        '    frozen_baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())\n',
    )
    source = replace_once(
        source,
        '        "device_execution": False,\n',
        '        "device_execution": False,\n'
        '        "performance_eligible": False,\n'
        '        "diagnostic_phase_clock": True,\n',
    )
    source = replace_once(
        source,
        "                entry(config=options)\n                torch.npu.synchronize()\n",
        '                with compilation_probe(side == "candidate", args.output / "instrumentation"):\n'
        "                    entry(config=options)\n                torch.npu.synchronize()\n",
    )
    start = source.index("                records = []\n")
    end = source.index('            save("complete")\n', start) + len('            save("complete")\n')
    source = (
        source[:start]
        + """                report["phase_windows"] = []
                save("diagnostic_phase_windows")
                for sample in range(3):
                    restore(fixture)
                    graphs["candidate"].replay()
                    stream.synchronize()
                    state, check = snapshot("candidate")
                    numerical = compare(state, expected)
                    report["checks"][f"phase_{sample}"] = {"guards": check, "numerical": numerical}
                    if not all(check.values()) or not all(v["status"] == "PASS" for v in numerical.values()):
                        raise AssertionError("Instrumented replay changed data or guards")
                    report["phase_windows"].append(export_entry(entries["candidate"], args.output / f"phase_{sample}"))
                    save()
            report["passed"] = True
            save("complete")
"""
        + source[end:]
    )
    (target / "run_pto.py").write_text(source)
    launcher = (here / "launch.sh").read_text()
    launcher = replace_once(
        launcher, 'harness="$here/runs/measurement_harness"', 'harness="$here/runs/phase_baseline_harness"'
    )
    # Keep the launcher at the experiment root: its `here` is used for artifacts.
    (here / "launch_phase.sh").write_text(launcher)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    assert "pypto.torch.begin_dfx()" not in source
    assert 'report["statistics_us"]' not in source and 'report["ratio_to_frozen"]' not in source
    subprocess.run(["bash", "-n", str(here / "launch_phase.sh")], check=True)
    report = dict(
        baseline="baseline.json",
        performance_eligible=False,
        purpose="Current SPMD stage endpoints and wait accumulation; three diagnostic replays",
        limitations="Probe Scalar/cache costs perturb scheduling; never use these windows for performance gains",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "phase_baseline_harness.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
