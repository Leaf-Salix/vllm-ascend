#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
mode=${1:?cpu or device}
attempt=${2:?fresh attempt name}
[[ "$attempt" =~ ^[a-zA-Z0-9_]+$ ]]
shift 2
probe_options=()
if [[ "$mode" == cpu ]]; then
    probe_options=(--cpu-only)
else
    [[ "$mode" == device ]]
    : "${TASK_DEVICE:?NPU execution requires task-submit}"
    if [[ ${1:-} == --device ]]; then
        [[ $# == 2 && "$2" == "$TASK_DEVICE" ]]
        shift 2
    fi
fi
[[ $# == 0 ]]
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
out="$here/runs/$attempt"
[[ ! -e "$out" ]]
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PYPTO_PROG_BUILD_DIR="$out/programs" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
export PYTHONPATH="$repo/tests/pypto_test:${PYTHONPATH:-}"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    "$here/score_gemv_pool_probe.py" --output "$out" "${probe_options[@]}"
