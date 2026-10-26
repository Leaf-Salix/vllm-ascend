#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
variant=${1:?spmd或mpmd}
case_name=${2:?固定场景}
attempt=${3:?独立输出名称}
[[ $# -ge 3 && $# -le 4 && "$attempt" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
extra=()
if [[ $# == 4 ]]; then
    [[ "$4" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
    extra+=(--snapshot "$here/snapshots/$4")
fi
out="$here/runs/$attempt"
[[ ! -e "$out" ]] || exit 2
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
cd "$repo"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$here/compile.py" --variant "$variant" --case "$case_name" --output "$out" "${extra[@]}"
