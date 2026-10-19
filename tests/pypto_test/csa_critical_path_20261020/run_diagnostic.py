# SPDX-License-Identifier: Apache-2.0
"""冻结输入目录并复用既有完整 CSA 全核阶段诊断，不采集正式性能。"""
import argparse
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
known, _ = parser.parse_known_args()
os.environ["PYPTO_PROG_BUILD_DIR"] = str(known.output.resolve() / "build_output")
os.environ["PYPTO_CACHE_DIR"] = str(known.output.resolve() / "cache")
log_path = known.output.resolve().parent / (known.output.name + "_device_logs")
log_path.mkdir(parents=True, exist_ok=True)
os.environ["ASCEND_PROCESS_LOG_PATH"] = str(log_path)
here = Path(__file__).resolve().parent
runner = here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/run_device.py"
sys.path.insert(0, str(runner.parent))
sys.argv[0] = str(runner)
runpy.run_path(str(runner), run_name="__main__")
