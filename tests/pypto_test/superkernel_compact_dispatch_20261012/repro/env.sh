#!/usr/bin/env bash
source "/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/base-env.snapshot.sh"
export SCALAR_OVERLAP_ROOT="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
export COMPACT_SDK_ROOT="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/sdk-compact-dispatch-20261012"
source "$COMPACT_SDK_ROOT/.venv/bin/activate"
export PTOAS_ROOT="$SCALAR_OVERLAP_ROOT/ptoas-build"
export PYTHONPATH="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto-compact-dispatch-20261012/python:/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/simpler-compact-dispatch-20261012/python:/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/simpler-compact-dispatch-20261012:/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/sdk-compact-dispatch-20261012/.venv/lib/python3.10/site-packages:/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/.venv/lib/python3.10/site-packages:/data/pyptouser/qinchuanyu/pto-eager/.venv-dsv4-0251rc1/lib/python3.10/site-packages:$SCALAR_OVERLAP_ROOT/vllm-ascend:/data/pyptouser/qinchuanyu/pto-eager/.cache/migration-v0.25.1rc1/vllm"
export ASCEND_PROCESS_LOG_PATH="$COMPACT_SDK_ROOT/artifacts/ascend"
export XDG_CACHE_HOME="$COMPACT_SDK_ROOT/.cache"
export PYPTO_CACHE_DIR="$COMPACT_SDK_ROOT/.cache/pypto"
export SIMPLER_CACHE_DIR="$COMPACT_SDK_ROOT/.cache/simpler"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYPTO_CODEGEN_MAX_WORKERS=1
export PYPTO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8
