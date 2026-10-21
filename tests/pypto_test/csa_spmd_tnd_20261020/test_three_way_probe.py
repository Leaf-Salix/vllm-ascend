# SPDX-License-Identifier: Apache-2.0
"""Check the diagnostic order and real shell snapshot before using an NPU."""

import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def test_six_windows_balance_every_entry_in_every_position():
    here = Path(__file__).resolve().parent
    path = here / "runs/three_way_harness_v116/tnd_performance.py"
    tree = ast.parse(path.read_text())
    profile = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "profile")
    sample_loop = next(n for n in ast.walk(profile) if isinstance(n, ast.For) and ast.unparse(n.target) == "sample")
    selection = sample_loop.body[0]
    assert isinstance(selection, ast.If) and ast.unparse(selection.test) == "extra_entries"
    entries = dict.fromkeys(("spmd_tnd", "spmd_original", "tnd_control_v106"))
    orders = []
    for sample in range(6):
        namespace = dict(entries=entries, extra_entries=True, sample=sample)
        exec(compile(ast.Module(body=[selection], type_ignores=[]), str(path), "exec"), namespace)
        orders.append(tuple(namespace["order"]))
    assert len(set(orders)) == 6
    for position in range(3):
        assert Counter(order[position] for order in orders) == dict.fromkeys(entries, 2)


def test_actual_snapshot_freezes_current_tnd_control(tmp_path):
    here = Path(__file__).resolve().parent
    harness = here / "runs/three_way_harness_v116"
    launcher = (harness / "run_probe.sh").read_text()
    marker = '"$baseline_source" "${manifest:-}" <<\'PY\'\n'
    script = launcher.split(marker, 1)[1].split("\nPY\n", 1)[0]
    snapshot = tmp_path / "snapshot"
    source = harness / "control_spmd"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(here.parents[2]),
            str(snapshot),
            str(harness / "run_probe.sh"),
            "probe",
            str(source),
            str(source),
            "",
        ],
        input=script,
        text=True,
        check=True,
    )
    manifest = json.loads((snapshot / "source_manifest.json").read_text())
    control = {k.removeprefix("control_spmd/"): v for k, v in manifest.items() if k.startswith("control_spmd/")}
    expected = json.loads((here / "borrowed_combined_cleanup_v106.json").read_text())["source_manifest"]
    assert all(control[name] == value for name, value in expected.items())
    assert all(hashlib.sha256((snapshot / path).read_bytes()).hexdigest() == value for path, value in manifest.items())
    assert (snapshot / "tnd_performance.py").read_bytes() == (harness / "tnd_performance.py").read_bytes()
