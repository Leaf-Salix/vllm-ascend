# SPDX-License-Identifier: Apache-2.0
"""只做无样例shape编译，保留完整错误；不初始化NPU。"""
import json,sys,traceback
from pathlib import Path
import dsv4_csa_env
dsv4_csa_env.activate()
from pypto import passes
from pypto.ir.printer import python_print
from pypto.runtime import CompileOptions
from pair_probe import pair_probe
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
r={"passed":False,"device_execution":False,"performance_measured":False}
try:
    (out/"symbolic.py").write_text(python_print(pair_probe.specialize()))
    artifact=pair_probe._resolve_kernel_artifact((),{"config":CompileOptions(platform="a2a3",output_dir=str(out/"kernel"),dump_passes=True,memory_planner=passes.MemoryPlanner.PYPTO)},allow_signature_mode=True)
    r.update(passed=True,artifact=str(artifact.output_dir))
except BaseException as exc:
    r.update(error=str(exc),traceback=traceback.format_exc())
finally:
    (out/"result.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in r.items() if k!='traceback'},ensure_ascii=False))
raise SystemExit(0 if r['passed'] else 1)
