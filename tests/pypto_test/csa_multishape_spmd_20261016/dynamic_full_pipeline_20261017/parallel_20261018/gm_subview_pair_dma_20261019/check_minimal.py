# SPDX-License-Identifier: Apache-2.0
"""CPU生成两个独立artifact，保存显式不支持错误与根Tensor对照。"""
import json
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()
from pypto import passes
from pypto.ir.printer import python_print
from pypto.runtime import CompileOptions
from minimal_repro import root_view, slice_view

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
reports = []
for name, entry in (("root", root_view), ("slice", slice_view)):
    folder = out / name
    folder.mkdir()
    result = {"case": name, "compiled": False, "device_execution": False}
    try:
        (folder / "symbolic.py").write_text(python_print(entry.specialize()))
        artifact = entry._resolve_kernel_artifact(
            (), {"config": CompileOptions(platform="a2a3", output_dir=str(folder / "kernel"),
                                            dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)},
            allow_signature_mode=True,
        )
        result.update(compiled=True, artifact=str(artifact.output_dir))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    reports.append(result)
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False), flush=True)
supported = all(row["compiled"] for row in reports)
(out / "result.json").write_text(json.dumps({"supported": supported, "cases": reports}, ensure_ascii=False, indent=2) + "\n")
raise SystemExit(0 if supported else 1)
