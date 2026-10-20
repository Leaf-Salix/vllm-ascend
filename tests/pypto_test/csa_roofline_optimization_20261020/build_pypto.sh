#!/usr/bin/env bash
# 仅在新独立PyPTO树构建；不复制扩展，不改共享安装。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
cd -- "$workspace/.worktrees/pypto-csa-topk-compact-20261020"
source .claude/skills/testing/load-env.sh
python - "$workspace" "$here" "$PYPTO_BUILD_JOBS" "$PYPTO_TEST_JOBS" <<'PYCODE'
import json,sys,subprocess
from pathlib import Path
root=Path(sys.argv[1]); worktree=Path.cwd(); cache=root/'.worktrees/pypto-csa-repro-20261019/build/CMakeCache.txt'
keys={'CMAKE_BUILD_TYPE','CMAKE_CXX_COMPILER','CMAKE_C_COMPILER','Python_EXECUTABLE','nanobind_DIR','PYPTO_BUILD_TORCH_NPU','PYPTO_BUILD_TORCH_NPU_TESTS','PYPTO_DEBUG_INFO_LEVEL','PYPTO_ACL_INCLUDE','PYPTO_ASCENDCL'}
args=['cmake','-S',str(worktree),'-B',str(worktree/'build')]
for line in cache.read_text().splitlines():
 if ':' in line and '=' in line and line.split(':')[0] in keys: args.append('-D'+line)
build=['cmake','--build',str(worktree/'build'),'--parallel',sys.argv[3]]
(Path(sys.argv[2])/'BUILD_COMMAND.json').write_text(json.dumps({'configure':args,'build':build,'build_jobs':sys.argv[3],'test_jobs':sys.argv[4],'binary_artifacts_copied':False},indent=2)+'\n')
subprocess.run(args,check=True)
subprocess.run(build,check=True)
PYCODE
