# SPDX-License-Identifier: Apache-2.0
"""从新PyPTO工作树执行有针对性的view/flatten回归，不占NPU。"""
import os
from pathlib import Path
import pytest
import pypto

assert Path(pypto.__file__).resolve().parents[2] == Path.cwd().resolve()
raise SystemExit(pytest.main([
    "tests/ut/ir/operators/test_tensor_ops.py",
    "tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py",
    "-k", "view or mat_load or padded_rank3", "-n", os.environ["PYPTO_TEST_JOBS"],
    "-q", "-o", "addopts=", "--junitxml=" + str(Path(os.environ["CSA_MULTIND_EVIDENCE"]) / "ut_v2.xml"),
]))
