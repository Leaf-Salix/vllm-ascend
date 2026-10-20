# SPDX-License-Identifier: Apache-2.0
"""采集冻结正式版的MemoryAccess/L2诊断，不把外层PMU时长作为L1性能。"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import memory_profile

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()
sys.modules["output_policy_matrix"] = memory_profile
try:
    runpy.run_path(str(here / "run_frozen_full.py"), run_name="__main__")
finally:
    result_path = args.output / "result.json"
    if result_path.exists():
        report = json.loads(result_path.read_text())
        report["performance_measured"] = False
        report["hardware_profile_collected"] = bool(report.get("source_matrix", {}).get("passed"))
        report["measurement_scope"] = "MemoryAccess及L2诊断；未采L1，不更新性能验收表"
        report["actual_matrix_callback"] = str(here / "memory_profile.py")
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (args.output / "actual_memory_profile.py").write_text((here / "memory_profile.py").read_text())
