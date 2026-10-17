# SPDX-License-Identifier: Apache-2.0
"""复用既有 CPU 编译入口，限定当前进程启用独立阶段的片上内存规划。"""

import runpy
import sys
from pathlib import Path

from shape_specialization import STATIC_ROOT
from phase_memory import scoped_memory_planning

output = Path(sys.argv[sys.argv.index("--output") + 1])
with scoped_memory_planning(output / "phase_memory"):
    runpy.run_path(str(STATIC_ROOT / "compile_jit_entry.py"), run_name="__main__")
