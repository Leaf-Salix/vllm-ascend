# SPDX-License-Identifier: Apache-2.0
"""在227无设备任务中验证未验收候选在NPU初始化前被拒绝。"""

import argparse
import json
import runpy
import sys
from pathlib import Path

import torch_npu

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
runner = Path(__file__).with_name("paired_single_layer.py")
sys.argv = [
    str(runner),
    str(args.root),
    str(args.output),
    "--candidate",
    str(args.root / "candidates/hc-padding-square-20261102-v1/source"),
]
try:
    runpy.run_path(str(runner), run_name="__main__")
except RuntimeError as error:
    assert "MIX members" in str(error), repr(error)
    assert not torch_npu.npu.is_initialized()
    result = {"passed": True, "rejected_before_npu_init": True, "reason": str(error)}
    (args.output / "guard-check.json").write_text(json.dumps(result) + "\n")
    print(json.dumps(result))
else:
    raise AssertionError("hardware guard did not reject")
