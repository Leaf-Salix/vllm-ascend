# SPDX-License-Identifier: Apache-2.0
"""沿用完整59组排序/同分回归，只把helper切换到成对压缩候选。"""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

import pypto

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()
logs = args.output.resolve().parent / (args.output.name + "_device_logs")
logs.mkdir(parents=True, exist_ok=True)
os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
pypto.configure_cache(pypto.CacheConfig(enabled=False))
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "topk_threshold_stages", here / "source_topk_paired/topk_threshold_stages.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.argv[0] = str(here / "topk_compact_probe.py")
runpy.run_path(sys.argv[0], run_name="__main__")
