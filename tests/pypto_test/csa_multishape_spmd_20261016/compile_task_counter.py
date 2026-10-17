# SPDX-License-Identifier: Apache-2.0
"""独立 CPU 编译任务计数候选，不触发设备执行。"""
import json
import runpy
import sys
from pathlib import Path
from shape_specialization import STATIC_ROOT
from task_completion import task_counter_codegen

output = Path(sys.argv[sys.argv.index("--output") + 1])
source = Path(sys.argv[sys.argv.index("--source") + 1])
layout = json.loads((source.parent / "layout.json").read_text())
with task_counter_codegen(output / "task_completion", layout["task_completion"]["phases"]):
    runpy.run_path(str(STATIC_ROOT / "compile_jit_entry.py"), run_name="__main__")
