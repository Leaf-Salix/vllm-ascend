# SPDX-License-Identifier: Apache-2.0
"""只验证本轮 gather_compare 类型/绑定/输出合同，不运行全仓无关测试。"""
import os
from pathlib import Path
import sys
import pytest
here = Path(__file__).resolve().parent
wt = here.parents[3] / ".worktrees/pypto-csa-topk-compact-20261020"
os.chdir(wt)
raise SystemExit(pytest.main(["tests/ut/ir/operators/test_gather_compare.py", "-q", "-n", sys.argv[1], "--junitxml=" + str(here / "pypto_ut_v3.xml")]))
