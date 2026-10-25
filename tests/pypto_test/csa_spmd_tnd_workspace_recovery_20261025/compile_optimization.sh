#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
candidate=$(realpath "${1:?候选目录}")
case_name=${2:?固定场景}
attempt=${3:?新输出名称}
[[ $# == 3 && "$attempt" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
out="$here/runs/$attempt"
[[ ! -e "$out" ]] || exit 2
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
export PYPTO_PROG_BUILD_DIR="$out/build_output"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
cd "$repo"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$here/frozen_optimization/run_pto.py" --cpu --compile --family tnd --case "$case_name" --candidate "$candidate" \
    --frozen "$here/frozen_optimization" --current-control "$here/candidates/internal_single_f72a" --candidate-init-layout fused --cases "$repo/tests/pypto_test/csa_performance_20261022/cases.json" \
    --output "$out" --uncompared-candidate
