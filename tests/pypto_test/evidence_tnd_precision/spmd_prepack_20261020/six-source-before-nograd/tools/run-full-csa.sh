#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
base=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex
source "$base/vector-mte2-20261019-codex/full-csa-live-staging-v2/env.sh"
export SPMD_EXPERIMENT="$base/precision-spmd-20261019"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE" PYTHONNOUSERSITE=1
export PYPTO_PROG_BUILD_DIR="$SPMD_EXPERIMENT/device-build" PYPTO_CACHE_DIR="$SPMD_EXPERIMENT/device-cache"
export CCACHE_DIR=/data/pyptouser/yejia/vllm-cann901-csa-0251rc1/src/pypto/.ccache
export CMAKE_BUILD_PARALLEL_LEVEL=4
out="$SPMD_EXPERIMENT/results/${1:?}"
mkdir -p "$(dirname "$out")"
test ! -e "$out"
export ASCEND_CACHE_PATH="$out-ascend-cache" ASCEND_PROCESS_LOG_PATH="$out-ascend-log"
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
ccache -s > "$out-ccache.log"
export ASCEND_OPP_PATH
ASCEND_OPP_PATH=$(python "$SOURCE_ROOT/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out-opp-env")
export VLLM_CACHE_ROOT="$out-vllm-cache"
timeout 750s python "$SPMD_EXPERIMENT/prepack-full-tools/probe_csa.py" --out "$out" --lengths "${2:?}" --layer "${3:-4}" --history "${4:-8191}" > "$out-run.log" 2>&1
