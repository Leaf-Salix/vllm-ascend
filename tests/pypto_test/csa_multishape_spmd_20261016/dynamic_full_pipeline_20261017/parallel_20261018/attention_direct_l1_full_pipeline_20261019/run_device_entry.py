# SPDX-License-Identifier: Apache-2.0
"""完整三方对照只替换当前进程的ISA头文件；原数学、精度门槛和性能口径沿旧harness。"""
import runpy
import sys
from pathlib import Path

import dsv4_csa_env
dsv4_csa_env.activate()
here = Path(__file__).resolve().parent
probe = here.parent / "attention_direct_l1_feasibility_20261019"
sys.path.insert(0, str(probe))
from private_isa_scope import install

output = Path(sys.argv[sys.argv.index("--output") + 1])
install(output.with_name(output.name + "_isa_audit"))
target = here.parent / "mgather_physical_plan_retest_20261019" / "run_device.py"
sys.path.insert(0, str(target.parent))
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
