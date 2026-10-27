#!/usr/bin/env bash
# 只补当前SDK下缺失的BSH两列或TND-MPMD两列。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(dirname -- "$repo")
family=${1:?bsh或tnd}
case_name=${2:?固定场景}
attempt=${3:?独立结果标签}
shift 3
if [[ ${1:-} == --device ]]; then
    [[ $# == 2 && "$2" == "${TASK_DEVICE:?}" ]] || exit 2
    shift 2
fi
[[ $# == 0 && "$attempt" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$family" == bsh || "$family" == tnd ]] || exit 2
: "${TASK_DEVICE:?必须由task-submit分配单张卡}"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
python - "$here/manifest.json" "$CSA_SHARED_TOOLCHAIN_MANIFEST" <<'PY'
import json,sys
frozen,current=[json.load(open(p)) for p in sys.argv[1:]]
assert frozen['shared_toolchain']==current and current['status']=='validated'
PY
out="$here/runs/${case_name}_${family}_${attempt}"
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
unset HC_PRE_TEST_OPERATOR_SOURCE
sides=(mpmd)
[[ "$family" == bsh ]] && sides=(spmd mpmd)
cd "$repo"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$here/sources/run_pto.py" --family "$family" --case "$case_name" \
    --candidate "$here/sources/tnd/pypto/deepseek_v4_flash_csa_tnd" \
    --frozen "$here/sources/$family" --cases "$here/cases.json" \
    --output "$out" --measure-sides "${sides[@]}" --device "$TASK_DEVICE"
