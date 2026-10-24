#!/usr/bin/env bash
set -eo pipefail
hca_source=$(realpath "${1:?冻结源码}")
hca_output=$(realpath -m "${2:?新的输出目录}")
shift 2
hca_record=$(dirname "$hca_source")/manifest.json
hca_repo=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$hca_record")
hca_workspace=$(dirname "$hca_repo")
source "$hca_workspace/env-dsv4-0251rc1.sh"
source "$hca_repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$hca_repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="${TASK_DEVICE:?经task-submit分配单卡}"
export LD_LIBRARY_PATH="$hca_repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$hca_source:$hca_source/tests/pypto_test:$PYTHONPATH"
export VLLM_ASCEND_ENABLE_NZ=2 PTO_CSA_VARIANT=performance PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
unset PTO_CSA_RING_HEAP_MB PTO_CSA_RING_TASK_WINDOW
export PYPTO_PROG_BUILD_DIR="$hca_output-build" PYPTO_CACHE_DIR="$hca_output-cache"
export ASCEND_WORK_PATH="$hca_output-ascend" ASCEND_PROCESS_LOG_PATH="$hca_output-ascend/log"
python "$hca_source/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$hca_workspace" --vllm-root "$hca_workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$hca_source/tests/pypto_test/hca_tnd_model_validation_20261024/single_card.py" "$hca_output" "$@"
