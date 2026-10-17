# SPDX-License-Identifier: Apache-2.0
"""CPU 编译全部原数学核的单任务组合，不启动 NPU。"""

import runpy
import sys
from pathlib import Path

from shape_specialization import STATIC_ROOT
from preserve_math_pipeline import preserved_pipeline


source = Path(sys.argv[sys.argv.index("--source") + 1])
output = Path(sys.argv[sys.argv.index("--output") + 1])
index = sys.argv.index("--dynamic-build")
dynamic_build = Path(sys.argv[index + 1])
del sys.argv[index:index + 2]
with preserved_pipeline(source.parent, dynamic_build, output / "preserved_all", all_stages=True):
    runpy.run_path(str(STATIC_ROOT / "compile_jit_entry.py"), run_name="__main__")
