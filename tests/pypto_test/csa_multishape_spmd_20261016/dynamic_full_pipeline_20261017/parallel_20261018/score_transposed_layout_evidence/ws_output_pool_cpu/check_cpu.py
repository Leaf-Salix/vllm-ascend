# SPDX-License-Identifier: Apache-2.0
import json,sys,traceback
from pathlib import Path
from pypto import passes
from pypto.runtime import CompileOptions
from pool_probe import pool_entry
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
r={"passed":False,"device_execution":False,"scope":"Acc子片集合后一次写回的编译探针"}
try:
    artifact=pool_entry._resolve_kernel_artifact((),{"config":CompileOptions(platform="a2a3",output_dir=str(out/"kernel"),dump_passes=True,memory_planner=passes.MemoryPlanner.PYPTO)},allow_signature_mode=True)
    r.update(passed=True,artifact=str(artifact.output_dir))
except BaseException as e:
    (out/"full_error.txt").write_text(traceback.format_exc())
    r["error"]=str(e)[:1200]
finally:
    (out/"result.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(r,ensure_ascii=False))
if not r["passed"]:raise SystemExit(1)
