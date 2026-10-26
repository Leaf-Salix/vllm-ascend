#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
candidate=$(realpath "${1:?冻结算子目录}")
case_name=${2:?固定场景}
attempt=${3:?新输出名称}
layout=${4:?初始化布局separate或fused}
shift 4
prepare_order=candidate-first
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-order)
            [[ $# -ge 2 ]] || exit 2
            prepare_order=$2
            shift 2
            ;;
        --device)
            [[ $# -ge 2 && "$2" == "${TASK_DEVICE:?通过task-submit提交}" ]] || exit 2
            shift 2
            ;;
        *) exit 2 ;;
    esac
done
[[ "$attempt" =~ ^[a-zA-Z0-9_]+$ && "$case_name" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$layout" == separate || "$layout" == fused || "$layout" == persistent_external ]] || exit 2
[[ "$prepare_order" == candidate-first || "$prepare_order" == reference-first ]] || exit 2
: "${TASK_DEVICE:?通过task-submit提交}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]] || exit 2
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/env.sh"
out="$here/runs/${case_name}_${attempt}"
[[ ! -e "$out" ]] || { printf '拒绝覆盖：%s\n' "$out" >&2; exit 2; }
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
export VLLM_CACHE_ROOT="${out}_vllm_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT"
cd "$repo"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$here/frozen/run_pto.py" --family tnd --case "$case_name" --candidate "$candidate" \
    --frozen "$here/frozen" --current-control "$here/candidates/current_f72a" --cases "$repo/tests/pypto_test/csa_performance_20261022/cases.json" \
    --output "$out" --uncompared-candidate --candidate-init-layout "$layout" --prepare-order "$prepare_order" --device "$TASK_DEVICE"
