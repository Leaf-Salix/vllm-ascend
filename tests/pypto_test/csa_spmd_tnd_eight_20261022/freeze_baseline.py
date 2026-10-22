# SPDX-License-Identifier: Apache-2.0
"""Audit existing eight-case evidence and preserve the measured combined_v1 bytes.

No device execution. The six-column report remains the immutable input; this
round measures improvements against its TND-SPMD implementation and samples.
"""

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def audit_case(case, row, expected_files):
    cell = row["TND-SPMD"]
    report = read(cell["source_report"])
    assert cell["status"] == "PASS" and report["passed"] and report["stage"] == "complete"
    assert report["case"] == case == cell["input_contract"]
    source = Path(report["sources"]["spmd"])
    actual_files = {p.name: digest(p.read_bytes()) for p in source.glob("*.py")}
    assert actual_files == expected_files
    for label, check in report["checks"].items():
        if label.startswith("spmd"):
            assert all(check["guards"].values()), label
            assert all(v["status"] == "PASS" for v in check["numerical"].values()), label
    windows = [w for w in report["windows"] if w["side"] == "spmd"]
    assert len(windows) >= 10 and len(windows) == cell["valid_windows"]
    values = []
    for window in windows:
        raw = read(window["raw"])
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1
        assert len(rows) == len({r[0] for r in rows}) == 72
        assert len({r[1] for r in rows}) == 1
        assert all(len(r) == 6 and r[4] >= r[3] > 0 for r in rows)
        assert raw["metadata"]["core_types"].count("aic") == 24
        assert raw["metadata"]["core_types"].count("aiv") == 48
        span = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
        assert math.isclose(span, window["incore_span_us"], abs_tol=1e-7)
        values.append(span)
    assert values == cell["samples_us"]
    assert statistics.median(values) == cell["median_us"]
    assert [w["raw"] for w in windows] == cell["raw"]
    return dict(case=case, source=str(source), source_report=cell["source_report"], cell=cell, validated=True)


def freeze(here):
    """Validate all input first, then write a new manifest and its necessary overlays."""
    repo = here.parents[2]
    previous = here.parent / "csa_spmd_tnd_opt_20261022"
    performance = here.parent / "csa_performance_20261022"
    report = read(performance / "PERFORMANCE.json")
    cases = read(performance / "cases.json")["cases"]
    combined = read(previous / "combined_v1.json")
    source = Path(combined["source"])
    files = {p.name: digest(p.read_bytes()) for p in source.glob("*.py")}
    assert files == combined["files"] and len(files) == 41
    assert len(cases) == 8 and [c["id"] for c in cases] == [f"C{i:02}" for i in range(1, 9)]
    audited = {c["id"]: audit_case(c, report["cells"][c["id"]], files) for c in cases}
    for row in report["cells"].values():
        for cell in row.values():
            if cell["status"] == "PASS":
                assert cell["valid_windows"] >= 10
                assert len(cell["samples_us"]) == cell["valid_windows"]
                assert statistics.median(cell["samples_us"]) == cell["median_us"]
            else:
                assert cell["status"] == "N/A"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    overlay = {}
    for name in files:
        committed = subprocess.check_output(["git", "show", f"{revision}:{package}/{name}"], cwd=repo)
        value = (source / name).read_bytes()
        if committed != value:
            overlay[name] = value
    frozen = dict(
        reference_kind="frozen_combined_v1_eight_20261022",
        revision=revision,
        package=package,
        overlay=sorted(overlay),
        files=files,
        source_evidence="../csa_performance_20261022/PERFORMANCE.json",
        evidence_sha256=digest((performance / "PERFORMANCE.json").read_bytes()),
        shared_toolchain=report["manifest"]["shared_toolchain"],
        execution=report["manifest"]["common_execution"],
        sampling=report["manifest"]["sampling"],
        cases=audited,
        performance_baseline="Existing ten-window medians; never replace with a slower rerun",
        comparison_checks="Preserve exact regression to the frozen implementation and original Native differences",
        optimization_target=dict(metric="equal_weight_geomean_candidate_over_frozen", case_count=8, maximum=0.9),
    )
    assert not (here / "baseline.json").exists()
    assert not (here / "baseline_overlay").exists()
    (here / "baseline_overlay").mkdir()
    for name, value in overlay.items():
        (here / "baseline_overlay" / (name + ".source")).write_bytes(value)
    with (here / "baseline.json").open("x") as stream:
        stream.write(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print(f"Audited 8 cases / 80 complete L1 windows; froze {len(files)} modules with {len(overlay)} overlays")


if __name__ == "__main__":
    freeze(Path(__file__).resolve().parent)
