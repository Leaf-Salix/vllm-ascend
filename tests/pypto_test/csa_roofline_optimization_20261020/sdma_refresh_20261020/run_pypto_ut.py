# SPDX-License-Identifier: Apache-2.0
"""只跑本次新增预取链路的相关 CPU 检查，完整 CSA 回归由原驱动负责。"""

import argparse
import os
from pathlib import Path

import pypto
import pytest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--jobs", type=int, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
root = Path(pypto.__file__).resolve().parents[2]
os.chdir(root)
raise SystemExit(pytest.main([
    "tests/ut/codegen/test_prefetch_codegen.py",
    "-q", "-n", str(args.jobs), "--junitxml=" + str(out / "pypto_ut.xml"),
]))
