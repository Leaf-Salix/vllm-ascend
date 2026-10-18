# SPDX-License-Identifier: Apache-2.0
"""pytest及xdist子进程均使用独立工作区的源码/扩展，不修改共享安装。"""

import os
import sys
from pathlib import Path


root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
loaded = sys.modules.get("pypto")
if loaded is not None and root not in Path(loaded.__file__).resolve().parents:
    raise RuntimeError("pytest在隔离插件之前载入了其它PyPTO，拒绝混用")
sys.meta_path[:] = [finder for finder in sys.meta_path
                   if "pypto.pypto_core" not in getattr(finder, "known_wheel_files", {})]
sys.path.insert(0, str(root / "python"))
import pypto
from pypto import pypto_core

assert root in Path(pypto.__file__).resolve().parents
assert root in Path(pypto_core.__file__).resolve().parents


def pytest_report_header():
    return f"isolated PyPTO: {pypto.__file__}; core: {pypto_core.__file__}"
