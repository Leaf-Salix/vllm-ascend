#!/usr/bin/env bash
# 从各自源码构建成对压缩工具链，不复制旧worktree二进制，不改共享安装。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
component=${1:?需要指定pypto或ptoas}
cd -- "$workspace/.worktrees/pypto-csa-pair-compact-20261020"
source .claude/skills/testing/load-env.sh
python - "$workspace" "$here" "$component" "$PYPTO_BUILD_JOBS" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root, output = map(Path, sys.argv[1:3])
component, jobs = sys.argv[3:5]
assert component in ("pypto", "ptoas")
worktree = root / f".worktrees/{component}-csa-pair-compact-20261020"
record = "BUILD_COMMAND.json" if component == "pypto" else "PTOAS_BUILD_COMMAND.json"
old = json.loads((output / record).read_text())
configure = [
    part.replace(f"{component}-csa-topk-compact-20261020",
                 f"{component}-csa-pair-compact-20261020")
    for part in old["configure"]
]
if component == "pypto":
    subprocess.run([
        "git", "-C", str(worktree), "-c", "protocol.file.allow=always", "-c",
        "submodule.runtime.url=" + str(root / ".worktrees/simpler-csa-core-args-256-20261016"),
        "submodule", "update", "--init", "runtime",
    ], check=True)
    configure.append("-DCMAKE_BUILD_PARALLEL_LEVEL=" + jobs)
build = ["cmake", "--build", str(worktree / "build"), "--parallel", jobs]
if component == "ptoas":
    build.extend(["--target", "ptoas-native"])
(output / f"PAIR_{component.upper()}_BUILD_COMMAND.json").write_text(
    json.dumps({"configure": configure, "build": build, "binary_copied": False},
               ensure_ascii=False, indent=2) + "\n"
)
subprocess.run(configure, check=True, cwd=worktree)
subprocess.run(build, check=True, cwd=worktree)
PY
