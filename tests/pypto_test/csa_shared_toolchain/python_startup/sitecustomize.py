# SPDX-License-Identifier: Apache-2.0
"""让服务子进程和测试进程使用同一源码，优先于历史 editable finder。"""

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

if os.environ.get("CSA_SHARED_PYPTO_ROOT") and os.environ.get("CSA_SHARED_SIMPLER_ROOT"):
    pypto = Path(os.environ["CSA_SHARED_PYPTO_ROOT"]).resolve()
    simpler = Path(os.environ["CSA_SHARED_SIMPLER_ROOT"]).resolve()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots = {
        "pypto": pypto / "python",
        "simpler": simpler / "python",
        "simpler_setup": simpler,
        "_task_interface": simpler / ".venv/lib" / version / "site-packages",
    }

    class SharedCSAToolchainFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            top = fullname.partition(".")[0]
            if top not in roots:
                return None
            search = path if "." in fullname else [str(roots[top])]
            spec = importlib.machinery.PathFinder.find_spec(fullname, search)
            if spec is None:
                raise ImportError(f"公共 CSA 工具链无法加载 {fullname}: {roots[top]}")
            return spec

    sys.meta_path.insert(0, SharedCSAToolchainFinder())
