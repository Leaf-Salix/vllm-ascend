#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
: "${TASK_DEVICE:?Queued device is required}"
snapshot=${1:?frozen diagnostic source directory}
out=${2:?new output directory}
shift 2
if [[ ${1:-} == --device ]]; then
    [[ $# == 2 && "$2" == "$TASK_DEVICE" ]] || exit 2
    shift 2
fi
[[ $# == 0 && ! -e "$out" ]] || exit 2
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
cd "$snapshot"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$snapshot/probe_scale_gather.py" --output "$out" --device "$TASK_DEVICE"
