# SPDX-License-Identifier: Apache-2.0
"""复用原三方完整 CSA 精度、状态、图重放和五窗 L1；不启用 SDMA。"""
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
logs = known.output.resolve().parent / (known.output.name + "_device_logs")
logs.mkdir(parents=True, exist_ok=True)
os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
base = Path(__file__).resolve().parent.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
runner = base / "parallel_20261018/mgather_physical_plan_retest_20261019/run_device.py"
sys.path.insert(0, str(runner.parent))
sys.argv[0] = str(runner)
runpy.run_path(str(runner), run_name="__main__")
