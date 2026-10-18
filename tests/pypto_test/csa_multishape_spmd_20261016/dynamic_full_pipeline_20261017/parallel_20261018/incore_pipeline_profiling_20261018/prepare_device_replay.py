#!/usr/bin/env python3
"""复用同一个独立kernel.so，另建Host入口验证ABI，不改正在模拟的用例。

只把Host设备号改为队列TASK_DEVICE，链接真实runtime。设备端二进制与
72核模拟使用的文件完全相同。输出放在独立目录，避免覆盖模拟器结果。
"""

import json
import os
import subprocess
from pathlib import Path

from prepare_standalone import HERE


def main():
    case = Path((HERE / "ACTIVE_CASE_PATH.txt").read_text().strip())
    build = case.parents[2] / "builds/dynamic_csa_full"
    dest = HERE / "build_output/device_replay_v2"
    dest.mkdir(parents=True, exist_ok=True)
    text = (case / "main.cpp").read_text()
    text = text.replace("int main() {", 'int main() {\n  const char* device_text = std::getenv("TASK_DEVICE");\n  if (!device_text) return 2;\n  const int device_id = std::atoi(device_text);')
    assert text.count("aclrtSetDevice(0)") == 1 and text.count("aclrtResetDevice(0)") == 1
    text = text.replace("aclrtSetDevice(0)", "aclrtSetDevice(device_id)").replace("aclrtResetDevice(0)", "aclrtResetDevice(device_id)")
    (dest / "main.cpp").write_text(text)
    toolkit = Path(os.environ["ASCEND_HOME_PATH"])
    isa = HERE.parents[6] / ".worktrees/simpler-csa-core-args-256-20261016/build/pto-isa"
    command = ["bisheng", "-xc++", "-std=c++17", "-O2", "-g", "-include", "stdint.h", "-include", "stddef.h",
               "-I"+str(isa/"tests/common"), "-I"+str(isa/"include"), "-I"+str(toolkit/"include"), str(dest/"main.cpp"),
               "-L"+str(build), "-L"+str(toolkit/"lib64"), "-Wl,--allow-shlib-undefined",
               "-Wl,-rpath,"+str(build), "-ldynamic_csa_full_msprof_kernel", "-lruntime", "-lstdc++", "-lascendcl",
               "-lm", "-ltiling_api", "-lplatform", "-lc_sec", "-ldl", "-lnnopbase", "-o", str(dest/"replay_device")]
    result = subprocess.run(command, capture_output=True, text=True)
    (dest/"build.json").write_text(json.dumps(dict(command=command, returncode=result.returncode,
                                                stdout=result.stdout, stderr=result.stderr,
                                                unchanged_kernel=str(build/"libdynamic_csa_full_msprof_kernel.so")),ensure_ascii=False,indent=2)+"\n")
    print(result.stdout, result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
