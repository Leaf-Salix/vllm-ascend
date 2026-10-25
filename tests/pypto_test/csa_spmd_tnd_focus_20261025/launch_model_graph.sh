#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 本轮实际服务入口验证；沿用已有框架图测试，不计性能。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
: "${TASK_DEVICE:?通过task-submit提交}"
if [[ ${1:-} == --device ]]; then
    [[ $# == 2 && "$2" == "$TASK_DEVICE" ]] || exit 2
    shift 2
fi
[[ $# == 0 ]] || exit 2
out="$here/runs/init_coeff_production_v1_model_graph"
[[ ! -e "$out" ]] || { printf '拒绝覆盖：%s\n' "$out" >&2; exit 2; }
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export LD_LIBRARY_PATH="$workspace/vllm-ascend-dsv4-pto-0251rc1/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$repo:$repo/tests/pypto_test:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 PTO_CSA_VARIANT=spmd_tnd PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
unset PTO_CSA_RING_HEAP_MB PTO_CSA_RING_TASK_WINDOW
export PYPTO_PROG_BUILD_DIR="$out-build" PYPTO_CACHE_DIR="$out-cache"
export ASCEND_WORK_PATH="$out-ascend" ASCEND_PROCESS_LOG_PATH="$out-ascend/log"
export ASCEND_CACHE_PATH="$out-ascend/cache"
mkdir -p "$PYPTO_PROG_BUILD_DIR" "$PYPTO_CACHE_DIR" "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
cd "$repo"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$workspace" --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$repo/tests/pypto_test/csa_tnd_model_validation_20261023/model_graph_probe.py" "$out" --variant spmd_tnd
