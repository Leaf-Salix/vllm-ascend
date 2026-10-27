#!/usr/bin/env bash
set -euo pipefail
align_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
source "$align_root/env-dsv4-0251rc1.sh"
source "$align_root/src/tests/pypto_test/csa_shared_toolchain/env.sh"
export PYPTO_PROG_BUILD_DIR="$align_root/build" PYPTO_CACHE_DIR="$align_root/cache"
export CCACHE_DIR="$CSA_SHARED_PYPTO_ROOT/.ccache" CMAKE_BUILD_PARALLEL_LEVEL=4
export HCCL_DETERMINISTIC=true PYTHONNOUSERSITE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
