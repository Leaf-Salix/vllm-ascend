# SPDX-License-Identifier: Apache-2.0
"""冻结公共源码并让配对两侧共享新版 SDMA context，沿用完整 CSA 验证与 L1 口径。"""

from pathlib import Path
import runpy
import sys

import pypto.torch


original_init = pypto.torch.init


def init_with_sdma(**kwargs):
    kwargs["enable_sdma"] = True
    return original_init(**kwargs)


pypto.torch.init = init_with_sdma
runner = Path(__file__).resolve().parent.parent / "run_frozen_full.py"
sys.path.insert(0, str(runner.parent))
sys.argv[0] = str(runner)
runpy.run_path(str(runner), run_name="__main__")
