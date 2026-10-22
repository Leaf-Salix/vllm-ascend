#!/usr/bin/env bash
# 单卡性能规范；调用方只选择场景和实现，不接受隐式算法覆盖。
set -eo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
workspace="$(dirname "$repo")"
family=${1:?native/tnd/bsh}
case_name=${2:?fixed case}
attempt=${3:?new attempt}
shift 3
if [[ ${1:-} == --device ]]; then
    [[ $# == 2 && "$2" == "${TASK_DEVICE:?}" ]] || exit 2
    shift 2
fi
[[ $# == 0 ]] || exit 2
: "${TASK_DEVICE:?必须经task-submit分配单张卡}"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
out="$here/runs/${case_name}_${family}_${attempt}"
test ! -e "$out" || { printf '拒绝覆盖：%s\n' "$out" >&2; exit 2; }
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
bootstrap=(python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"
    --workspace-root "$workspace" --shared-toolchain
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm")
if [[ "$family" == native ]]; then
    mkdir -p "$out/compile"
    export ASCEND_OPP_PATH="$(python "$repo/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/private_opp")"
    read -r history batch < <(python - "$here/cases.json" "$case_name" <<'PY'
import json,sys
c=next(c for c in json.load(open(sys.argv[1]))['cases'] if c['name']==sys.argv[2])
print(c['history_label'],c['active_requests'])
PY
)
    cd "$out/compile"
    exec "${bootstrap[@]}" "$here/native_sources_v2/run_native.py" --case "$case_name" --history "$history" --batch "$batch" --output "$out" --device "$TASK_DEVICE"
elif [[ "$family" == tnd || "$family" == bsh ]]; then
    cd "$repo"
    runner="$here/sources/run_pto.py"
    frozen="$here/sources"
    if [[ "$family" == bsh && "$case_name" == h131072_b1_s6 ]]; then runner="$here/bsh_sources_v2/run_pto.py"; fi
    if [[ "$family" == bsh && "$case_name" == h8192_b32_s6 ]]; then
        runner="$here/bsh_short_sources/run_pto.py"
        frozen="$here/bsh_short_sources"
    fi
    exec "${bootstrap[@]}" "$runner" --family "$family" --case "$case_name" \
        --candidate "$here/sources/pypto/deepseek_v4_flash_csa_tnd" --frozen "$frozen" \
        --cases "$here/cases.json" --output "$out" --device "$TASK_DEVICE"
else
    exit 2
fi
