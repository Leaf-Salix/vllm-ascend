# SPDX-License-Identifier: Apache-2.0
"""在独立父包内加载冻结算子，确保相对依赖也来自同一快照。"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def load_snapshot(path: Path, label: str):
    importlib.import_module("vllm_ascend.ops.pypto")
    name = "vllm_ascend.ops.pypto._tnd_nz_" + label
    if name in sys.modules:
        raise RuntimeError(f"冻结命名空间已加载：{name}")
    module = types.ModuleType(name)
    module.__path__ = [str(path.resolve())]
    module.__package__ = name
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = module
    return name
