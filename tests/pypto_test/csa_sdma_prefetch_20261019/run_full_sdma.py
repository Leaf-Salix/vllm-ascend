# SPDX-License-Identifier: Apache-2.0
"""沿用现有三方完整 CSA 验证和 L1 计时，仅显式启用新运行时的 SDMA。

所有对照均使用同一新工具链、同一个 SDMA-enabled context；参照算子没有
预取指令。这样不会将初始化能力开关或新旧编译器差别记为候选预取收益。
"""

import argparse
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
known, _ = parser.parse_known_args()
# 旧 run_device.sh 负责设置这些目录。独立工具链入口不经过该 shell，
# 因此在导入/编译前显式恢复同样的隔离，避免多个任务共用 build_output。
os.environ["PYPTO_PROG_BUILD_DIR"] = str(known.output.resolve() / "build_output")
os.environ["PYPTO_CACHE_DIR"] = str(known.output.resolve() / "cache")
os.environ.pop("HC_PRE_TEST_OPERATOR_SOURCE", None)

import pypto.torch


original_init = pypto.torch.init


def init_with_sdma(**kwargs):
    kwargs["enable_sdma"] = True
    return original_init(**kwargs)


pypto.torch.init = init_with_sdma
here = Path(__file__).resolve().parent
runner = (here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
          / "parallel_20261018/mgather_physical_plan_retest_20261019/run_device.py")
sys.path.insert(0, str(runner.parent))
sys.argv[0] = str(runner)
runpy.run_path(str(runner), run_name="__main__")
