# SPDX-License-Identifier: Apache-2.0
"""复用冻结完整CSA驱动，只在测试进程替换诊断回调；不改正式算子和安装。"""

import argparse
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profile_pair

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path, required=True)
known, _ = parser.parse_known_args()
# 原驱动会复制原矩阵脚本作为历史附件；实际回调另存本诊断脚本，防止误读。
sys.modules["output_policy_matrix"] = profile_pair
try:
    runpy.run_path(str(HERE.parent / "run_full.py"), run_name="__main__")
finally:
    result_path = known.output / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result["performance_measured"] = False
        result["hardware_profile_collected"] = bool(
            result.get("source_matrix", {}).get("passed", False))
        result["measurement_scope"] = "只采外部PipeUtilization，未采L1、不用于性能验收或更新正式性能表"
        result["actual_matrix_callback"] = str(HERE / "profile_pair.py")
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        for name in ("profile_pair.py", "run_profile.py"):
            (known.output / ("pmu_" + name)).write_text((HERE / name).read_text())
