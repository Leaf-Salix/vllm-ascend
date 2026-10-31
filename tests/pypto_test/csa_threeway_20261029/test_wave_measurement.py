# SPDX-License-Identifier: Apache-2.0
"""227 CPU验证观测副本，不能把空计时或跨入场轨迹当作公平性能。"""

import ast
import hashlib
import json
from pathlib import Path

import pytest
from csa_threeway_20261029.wave_measurement import prepare_wave_runner


@pytest.mark.parametrize("pto", [False, True])
def test_wave_runner_preserves_frozen_sources(tmp_path, pto):
    helper = tmp_path / "source/tests/pypto_test/csa_threeway_20261029"
    helper.mkdir(parents=True)
    source = helper.parents[2]
    observer = helper / "observer.py"
    observer.write_bytes(Path(__file__).with_name("observer.py").read_bytes())
    mixed = source / "tests/pypto_test/low_acceptance_20261001/mixed.py"
    mixed.parent.mkdir()
    mixed.write_text(
        "def run_batches():\n    records = []\n"
        "        outputs = generate_aligned_batch(llm, prompts, params)\n"
        "    row = {'sampling': {'temperature': 0}}\n"
    )
    # Use syntactically valid indentation while retaining the exact production insertion point.
    mixed.write_text(mixed.read_text().replace("    records = []\n", "    records = []\n    for chunk in chunks:\n"))
    runner = tmp_path / "runner.py"
    runner.write_text(
        'worker = "csa_threeway_20261029.observer.TimedObserver"\n'
        "from low_acceptance_20261001.mixed import run_batches\n"
    )
    originals = {p: p.read_bytes() for p in (observer, mixed, runner)}
    manifest = {
        "files": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (observer, mixed)}
    }
    (source.parent / "manifest.json").write_text(json.dumps(manifest))
    audit = None
    if pto:
        audit_file = helper / "runtime_test.py"
        audit_file.write_text("from .observer import TimedObserver\nclass RuntimeAuditObserver(TimedObserver): pass\n")
        runner.write_text(runner.read_text().replace("observer.TimedObserver", "runtime_test.RuntimeAuditObserver"))
        originals[runner] = runner.read_bytes()
        audit = {"observer": str(audit_file)}
    result, receipt = prepare_wave_runner(source, runner, audit)
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert prepare_wave_runner(source, runner, audit) == (result, receipt)
    for row in receipt.values():
        text = Path(row["path"]).read_text()
        ast.parse(text)
        assert hashlib.sha256(text.encode()).hexdigest() == row["sha256"]
    updated = Path(receipt["wave_observer"]["path"]).read_text()
    assert "metadata.decode.num_reqs_actual == 8" in updated
    assert "metadata.decode.num_reqs_actual == 40" not in updated
    generation = Path(receipt["wave_mixed"]["path"]).read_text()
    assert generation.index("time.perf_counter() - begin") > generation.index("outputs = generate_aligned_batch")
    observer.write_text("tampered\n")
    with pytest.raises(AssertionError):
        prepare_wave_runner(source, runner, audit)
