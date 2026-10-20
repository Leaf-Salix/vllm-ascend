#!/usr/bin/env bash
# 独立重建新版 core 与 torch adapter，不复制任何旧二进制。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
cd -- "$workspace/.worktrees/pypto-csa-sdma-refresh-20261020"
source .claude/skills/testing/load-env.sh
python - "$workspace" "$here" "$PYPTO_BUILD_JOBS" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root, output = map(Path, sys.argv[1:3])
worktree = Path.cwd()
cache = root / ".worktrees/pypto-csa-sdma-20261019/build/CMakeCache.txt"
# 配套 runtime 提交仅在本地，不能让自动子模块初始化去远端请求不存在的提交。
# 命令级 URL 覆盖不会修改其它 worktree 的配置，也不需要提前 push。
subprocess.run([
    "git", "-c", "protocol.file.allow=always", "-c",
    "submodule.runtime.url=" + str(root / ".worktrees/simpler-csa-sdma-refresh-20261020"),
    "submodule", "update", "--init", "runtime",
], check=True)
keys = {
    "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER",
    "Python_EXECUTABLE", "nanobind_DIR", "PYPTO_BUILD_TORCH_NPU",
    "PYPTO_BUILD_TORCH_NPU_TESTS", "PYPTO_DEBUG_INFO_LEVEL",
    "PYPTO_ACL_INCLUDE", "PYPTO_ASCENDCL",
}
configure = ["cmake", "-S", str(worktree), "-B", str(worktree / "build")]
# 当前 libbacktrace.cmake 读取 CMake 变量，而非 build --parallel 或同名环境变量。
# 显式配置该值，避免外部 make 退化成不带数字的 -j。
configure.append("-DCMAKE_BUILD_PARALLEL_LEVEL=" + sys.argv[3])
for line in cache.read_text().splitlines():
    if ":" in line and "=" in line and line.split(":")[0] in keys:
        configure.append("-D" + line)
build = ["cmake", "--build", str(worktree / "build"), "--parallel", sys.argv[3]]
(output / "PYPTO_BUILD_COMMAND.json").write_text(
    json.dumps({"configure": configure, "build": build}, indent=2) + "\n"
)
subprocess.run(configure, check=True)
subprocess.run(build, check=True)
PY
