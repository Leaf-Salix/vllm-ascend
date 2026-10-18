#!/usr/bin/env python3
"""采集异常时的诊断：同一72核binary在相同camodel中执行，暂不注入profiler。

必须经CPU队列；不占NPU、不改数学、不减少核。此结果用于区分模型执行与
profiler采集问题，不是incore trace，也绝不能当作性能结果。
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from prepare_standalone import HERE


def main():
    case = Path((HERE / "ACTIVE_CASE_PATH.txt").read_text().strip())
    run = case.parents[2]
    build = run / "builds/dynamic_csa_full"
    originals = list((run / "funcs/dynamic_csa_full/collect/out").glob("*/device0/tmp_dump/config"))
    assert len(originals) == 1
    dest = HERE / "build_output/model_without_profile_v1"
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copytree(originals[0], dest / "config")
    (dest / "model_logs").mkdir()
    toolkit = Path(os.environ["ASCEND_HOME_PATH"])
    env = os.environ.copy()
    env.update(CAMODEL_CONFIG_PATH=str(dest / "config"), CAMODEL_LOG_PATH=str(dest / "model_logs"),
               CAMODEL_SOC_VERSION="Ascend910_9392", ACL_DEVICE_ID="0", LD_PRELOAD="libruntime_camodel.so")
    env["LD_LIBRARY_PATH"] = str(toolkit / "aarch64-linux/simulator/dav_2201/lib") + ":" + str(build) + ":" + env.get("LD_LIBRARY_PATH", "")
    started = time.monotonic()
    with (dest / "model.log").open("w") as output:
        result = subprocess.run([str(build / "dynamic_csa_full_msprof_sim")],cwd=dest,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=600)
    report = dict(returncode=result.returncode, diagnostic_wall_seconds=time.monotonic()-started,
                  unchanged_binary=str(build / "dynamic_csa_full_msprof_sim"), cores="24 AIC + 48 AIV",
                  profiling_injection=False, task_id=os.environ.get("TASK_ID"), performance_result=False)
    (HERE / "MODEL_WITHOUT_PROFILE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    if result.returncode:
        return result.returncode
    return subprocess.run([os.sys.executable,str(HERE / "validate_replay.py"),"--case",str(dest),
                           "--output",str(HERE / "MODEL_WITHOUT_PROFILE_VALIDATION.json")],check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
