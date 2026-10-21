# SPDX-License-Identifier: Apache-2.0
"""CPU checks that incremental measurements cannot become BSH gate evidence."""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from evaluate_s6_cohort import evaluate


@pytest.mark.parametrize("marker", ["result", "snapshot"])
def test_incremental_measurements_rejected_by_bsh_gate(tmp_path, marker):
    directory = tmp_path / "measured"
    directory.mkdir()
    report = {"comparison_reference": {"kind": "adopted_tnd_output_balance_v4"}} if marker == "result" else {}
    (directory / "result.json").write_text(json.dumps(report))
    if marker == "snapshot":
        snapshot = tmp_path / "measured_source"
        snapshot.mkdir()
        (snapshot / "comparison_reference.json").write_text("{}")
    task_ids = [f"task_{i}" for i in range(8)]
    jobs = [{"task_id": task_id, "output": str(directory)} for task_id in task_ids]
    with pytest.raises(ValueError, match="incremental TND comparison cannot replace"):
        evaluate(jobs, task_ids)


@pytest.mark.parametrize(
    "kind,version,expected_kind",
    [
        ("qa", "v93b", "adopted_tnd_output_balance_v4"),
        ("ws", "v93b", "adopted_tnd_output_balance_v4"),
        ("combined", "v108b", "adopted_tnd_borrowed_combined_v106"),
        ("qb_wide", "v112", "validated_qb_wide_v108b"),
        ("shared", "v118", "adopted_tnd_borrowed_combined_v106"),
    ],
)
def test_actual_launcher_snapshot_uses_frozen_tnd_reference(tmp_path, kind, version, expected_kind):
    here = Path(__file__).resolve().parent
    harness = here / (
        f"runs/incremental_shared_{version}" if kind == "shared" else f"runs/incremental_{kind}_harness_{version}"
    )
    if not harness.exists():
        pytest.skip("Requires the local frozen incremental probe; skip is not device evidence")
    info = json.loads((here / (harness.name + ".json")).read_text())
    original = Path(info["original_harness"])
    for name in info["unchanged_checks_and_timing"]:
        assert (harness / name).read_bytes() == (original / name).read_bytes()
    launcher = (harness / "run_probe.sh").read_text()
    marker = '"$baseline_source" "${manifest:-}" <<\'PY\'\n'
    script = launcher.split(marker, 1)[1].split("\nPY\n", 1)[0]
    snapshot = tmp_path / "snapshot"
    reference = harness / "reference"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(here.parents[2]),
            str(snapshot),
            str(harness / "run_probe.sh"),
            "probe",
            str(reference),
            str(reference),
            "unused_bsh_manifest",
        ],
        input=script,
        text=True,
        check=True,
    )
    assert not (snapshot / "bsh_baseline_manifest.json").exists()
    saved = json.loads((snapshot / "comparison_reference.json").read_text())
    assert saved["kind"] == expected_kind
    if kind == "qb_wide":
        assert saved["adopted"] is False
    assert saved["formal_bsh_gate_eligible"] is False
    for name, expected in saved["files"].items():
        assert hashlib.sha256((snapshot / "original_spmd" / name).read_bytes()).hexdigest() == expected
    for name in info["unchanged_checks_and_timing"]:
        assert (snapshot / name).read_bytes() == (harness / name).read_bytes()
    if kind == "ws":
        assert (snapshot / "scale_layout_diagnostics.py").read_bytes() == (
            harness / "scale_layout_diagnostics.py"
        ).read_bytes()


def test_wrong_reference_cannot_be_mislabeled_as_adopted_tnd(tmp_path):
    here = Path(__file__).resolve().parent
    output = tmp_path / "private_harness"
    result = subprocess.run(
        [
            sys.executable,
            str(here / "prepare_incremental_harness.py"),
            "--reference",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "reserved for adopted output_balance v4" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "source,accepted",
    [
        ("PTOAS__WAIT_GE_BACKOFF<150>(signal);", True),
        ("PTOAS__WAIT_GE_BACKOFF<100>(signal);", False),
        ("pto::comm::TWAIT(signal);", False),
        ("PTOAS__WAIT_GE_BACKOFF<150>(signal); pto::comm::TWAIT(signal);", False),
    ],
)
def test_incremental_callback_requires_native150(source, accepted):
    here = Path(__file__).resolve().parent
    path = here / "runs/incremental_qa_harness_v93b/bsh_probe.py"
    if not path.exists():
        pytest.skip("Requires the local frozen incremental probe")
    main = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    callbacks = [node for node in main.body if isinstance(node, ast.FunctionDef)]
    namespace = {"re": re, "args": SimpleNamespace(wait_first_fast_polls=0)}
    exec(compile(ast.Module(body=callbacks, type_ignores=[]), str(path), "exec"), namespace)
    callback = namespace["rewrite_or_accept_native"]
    if accepted:
        actual, audit = callback(source)
        assert actual == source
        assert audit["native_lowering"] and not audit["source_rewritten"]
        assert audit["backoff_syscnt_ticks"] == 150
    else:
        with pytest.raises(ValueError):
            callback(source)


@pytest.mark.parametrize(
    "enabled,values,expected",
    [
        (True, (37, 37), True),
        (True, (37, 38), False),
        (False, None, None),
    ],
)
def test_unused_score_arena_guard(enabled, values, expected):
    here = Path(__file__).resolve().parent
    path = here / "runs/short_half_ub_harness_v95/run_probe.py"
    if not path.exists():
        pytest.skip("Requires the local frozen short-half probe")
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "workspace_padding_checks"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)

    class HostValues:
        def cpu(self):
            assert values is not None, "Disabled branch must not read the score arena"
            return self

        def __eq__(self, other):
            return SimpleNamespace(all=lambda: all(value == other for value in values))

    entry = SimpleNamespace(
        policy=SimpleNamespace(score_tile=768, hc_consumer_reduce=False, short_half_score_ub=enabled),
        buffers={"indexer_transfer": SimpleNamespace(ndim=2, shape=(96, 768)), "score_arena": HostValues()},
    )
    checks = namespace["workspace_padding_checks"](entry)
    assert checks.get("score_arena_unused_untouched") is expected
