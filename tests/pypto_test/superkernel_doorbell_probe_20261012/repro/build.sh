#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
cd "$p/pypto"
source "$p/env.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$p/artifacts/boundary_doorbell_probe_20261012/repro/driver.py" build --root "$p/artifacts/boundary_doorbell_probe_20261012"
