# SPDX-License-Identifier: Apache-2.0
"""只检查直接 L1 采集与三槽动态生命周期的完整编译，不报告设备或性能通过。"""
import json
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
from pypto import passes
from pypto.ir.printer import python_print
from pypto.runtime import CompileOptions
from triple_slot_probe import triple_slot

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
report = dict(compiled=False, device_execution=False, full_csa_validated=False)
try:
    (out / "symbolic.py").write_text(python_print(triple_slot.specialize()))
    artifact = triple_slot._resolve_kernel_artifact(
        (),
        {"config": CompileOptions(platform="a2a3", output_dir=str(out / "kernel"),
                                  dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)},
        allow_signature_mode=True,
    )
    report.update(compiled=True, artifact=str(artifact.output_dir))
except Exception as exc:
    report.update(error=str(exc), traceback=traceback.format_exc())
(out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)
raise SystemExit(0 if report["compiled"] else 1)
