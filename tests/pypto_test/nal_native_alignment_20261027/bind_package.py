# SPDX-License-Identifier: Apache-2.0
"""把实验PTO包与冻结Native插件分开导入。"""

import sys
import types
from pathlib import Path


def bind():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/pypto"
    package = types.ModuleType("nal_alignment")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return source
