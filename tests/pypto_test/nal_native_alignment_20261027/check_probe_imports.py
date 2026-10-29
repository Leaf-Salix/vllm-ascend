# SPDX-License-Identifier: Apache-2.0
"""不初始化NPU，复核完整观测脚本的实际顶层导入顺序。"""

import ast
import json
import os
import sys
from pathlib import Path

import pypto
import simpler
import torch_npu

assert not torch_npu.npu.is_initialized()
target = Path(__file__).with_name("probe_spmd.py")
tree = ast.parse(target.read_text(), filename=str(target))
prefix = []
for statement in tree.body:
    if isinstance(statement, ast.Assign) and any(
        isinstance(name, ast.Name) and name.id == "out" for name in statement.targets
    ):
        break
    prefix.append(statement)
scope = {"__name__": "probe_import_check", "__file__": str(target)}
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(target), "exec"), scope)
assert not torch_npu.npu.is_initialized()
for module, key in ((pypto, "CSA_SHARED_PYPTO_ROOT"), (simpler, "CSA_SHARED_SIMPLER_ROOT")):
    assert Path(module.__file__).is_relative_to(Path(os.environ[key]))
assert os.environ["PTOAS_ROOT"] == os.environ["CSA_SHARED_PTOAS_ROOT"] and os.environ["ASCEND_HOME_PATH"]
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "pass": True,
            "npu_initialized": False,
            "pypto": pypto.__file__,
            "simpler": simpler.__file__,
            "PTOAS_ROOT": os.environ["PTOAS_ROOT"],
            "ASCEND_HOME_PATH": os.environ["ASCEND_HOME_PATH"],
        }
    )
    + "\n"
)
