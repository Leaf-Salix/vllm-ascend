#!/usr/bin/env python3
"""Score硬件归因入口的符号编译；不供应实值shape，不初始化设备。"""

import json
import sys
import traceback
from pathlib import Path

from pypto import passes
from pypto.runtime import CompileOptions
from score_kernel import score_entry


output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=False)
controls = dict(G=6, N=64, STEP=1024, PREFETCH=1, TO_L0=True, BALANCED=True)
report = {"passed": False, "controls": controls, "sample_shapes_supplied": False,
          "device_execution": False, "performance_measured": False}
try:
    artifact = score_entry._resolve_kernel_artifact(
        (), {**controls, "config": CompileOptions(
            platform="a2a3", output_dir=str(output / "kernel"), dump_passes=True,
            memory_planner=passes.MemoryPlanner.PYPTO)}, allow_signature_mode=True,
    )
    report.update(passed=True, artifact=str(artifact.output_dir))
except BaseException as error:
    report.update(error=str(error), traceback=traceback.format_exc())
    raise
finally:
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("passed", "artifact", "error")}, ensure_ascii=False))
