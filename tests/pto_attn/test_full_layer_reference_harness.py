# SPDX-License-Identifier: Apache-2.0
"""Regression for candidate imports preceding the reference environment gate."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("reject_environment", [False, True])
def test_environment_gate_precedes_candidate_import(tmp_path, reject_environment):
    reference = tmp_path / "reference"
    tests = reference / "tests/pypto_test"
    tests.mkdir(parents=True)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "__init__.py").write_text("")
    marker = tmp_path / "candidate-imported"
    activation = "raise RuntimeError('source rejected')" if reject_environment else "ready = True"
    (tests / "dsv4_csa_env.py").write_text(
        "ready = False\ndef activate():\n    global ready\n    " + activation + "\n    return 'reference'\n"
    )
    (candidate / "decode_csa.py").write_text(
        "from pathlib import Path\nimport dsv4_csa_env\n"
        "assert dsv4_csa_env.ready\n"
        f"Path({str(marker)!r}).write_text('loaded after gate')\n"
    )
    (tests / "dsv4_csa_single_layer.py").write_text(
        "import sys\nfrom dsv4_csa_env import activate\n"
        "assert activate() == 'reference'\n"
        "assert sys.argv[1:] == ['--variant', 'performance', '--batch', '4']\n"
        "assert sys.modules['vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa'] "
        "is sys.modules['leaf_dspark_layer.decode_csa']\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run_full_layer_reference.py")),
            "--reference-checkout",
            str(reference),
            "--candidate-package",
            str(candidate),
            "--batch",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if reject_environment:
        assert result.returncode != 0
        assert "source rejected" in result.stderr
        assert not marker.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "loaded after gate"
