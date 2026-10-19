# SPDX-License-Identifier: Apache-2.0
"""复用既有72核阶段时戳诊断，不把插桩时长作为正式性能。

一次完整CSA调用读取各核各阶段发布时刻与累计等待；正式性能仍来自
run_full_sdma.py的无阶段插桩配对L1。仅用于解释预取延长了哪段依赖链。
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
          / "run_device.py")
sys.path.insert(0, str(runner.parent))
sys.argv[0] = str(runner)
runpy.run_path(str(runner), run_name="__main__")
