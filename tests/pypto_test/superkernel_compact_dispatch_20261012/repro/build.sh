#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
cd "$p/pypto"
source "$p/env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
a="$p/artifacts/boundary_compact_dispatch_20261012"
python "$a/repro/driver.py" build --root "$a" > "$a/build.log" 2>&1
python "$a/repro/driver.py" compile --root "$a" --compiled "$a/compiled" --modes plain early --scalar-steps 256 > "$a/compile.log" 2>&1
