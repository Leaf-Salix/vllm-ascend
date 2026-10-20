# SPDX-License-Identifier: Apache-2.0
"""复用原59组微验证，仅将被测helper绑定为容量缩短候选。"""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

import pypto

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
known, _ = parser.parse_known_args()
logs = known.output.resolve().parent / (known.output.name + "_device_logs")
logs.mkdir(parents=True, exist_ok=True)
os.environ["ASCEND_PROCESS_LOG_PATH"] = str(logs)
# 同一算子源需要验证不同PTOAS修复；禁止复用修复前的生成物。
pypto.configure_cache(pypto.CacheConfig(enabled=False))

here = Path(__file__).resolve().parent
source = here / "source_topk_payload/topk_threshold_stages.py"
spec = importlib.util.spec_from_file_location("topk_threshold_stages", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runner = here / "topk_compact_probe.py"
sys.argv[0] = str(runner)
runpy.run_path(str(runner), run_name="__main__")
