# SPDX-License-Identifier: Apache-2.0
"""设备任务之前核验Q诊断所需的Native导入顺序，不初始化NPU。"""

import importlib
import json
import sys
from pathlib import Path

import torch_npu

assert not torch_npu.npu.is_initialized()
importlib.import_module("vllm_ascend.ops")
operator = importlib.import_module("vllm_ascend.attention.dsa_v1").DeviceOperator
assert not torch_npu.npu.is_initialized()
receipt = {"pass": True, "npu_initialized": False, "native_operator": operator.apply_dsa_q_rms.__qualname__}
Path(sys.argv[1]).write_text(json.dumps(receipt) + "\n")
