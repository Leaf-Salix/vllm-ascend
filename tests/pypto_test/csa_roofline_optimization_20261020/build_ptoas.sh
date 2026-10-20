#!/usr/bin/env bash
# 独立构建，不复用其它 worktree 的二进制；沿用已验证 LLVM 和本机并行限制。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/.worktrees/pypto-csa-topk-compact-20261020/.claude/skills/testing/load-env.sh"
python - "$workspace" "$here" "$PYPTO_BUILD_JOBS" <<'PYCODE'
import json,sys,subprocess
from pathlib import Path
root=Path(sys.argv[1]); here=Path(sys.argv[2]); wt=root/'.worktrees/ptoas-csa-topk-compact-20261020'
cache=root/'.worktrees/ptoas-csa-repro-20261019/build/CMakeCache.txt'
keys={'CANN_3RD_LIB_PATH','CMAKE_BUILD_TYPE','CMAKE_CXX_COMPILER','CMAKE_C_COMPILER','LLVM_DIR','MLIR_DIR','PTO_ENABLE_PYTHON_BINDING','PTO_ENABLE_VFSIM_COSTMODEL'}
args=['cmake','-G','Ninja','-S',str(wt),'-B',str(wt/'build')]
for line in cache.read_text().splitlines():
 if ':' in line and '=' in line and line.split(':')[0] in keys: args.append('-D'+line)
build=['cmake','--build',str(wt/'build'),'--target','ptoas-native','--parallel',sys.argv[3]]
(here/'PTOAS_BUILD_COMMAND.json').write_text(json.dumps({'configure':args,'build':build},indent=2)+'\n')
subprocess.run(args,check=True)
subprocess.run(build,check=True)
PYCODE
