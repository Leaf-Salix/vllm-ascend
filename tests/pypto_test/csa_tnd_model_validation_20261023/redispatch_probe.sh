#!/usr/bin/env bash
set -eo pipefail
source_tree=$(realpath "${1:?冻结源码}")
output=$(realpath -m "${2:?输出目录}")
record=$(dirname "$source_tree")/manifest.json
repo=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$record")
workspace=$(dirname "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="${TASK_DEVICE:?单卡队列分配}"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_tree:$source_tree/tests/pypto_test:$PYTHONPATH"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export ASCEND_WORK_PATH="$output-ascend" ASCEND_PROCESS_LOG_PATH="$output-ascend/log"
export ASCEND_CACHE_PATH="$output-ascend/cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
python "$source_tree/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$workspace" --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$source_tree/tests/pypto_test/csa_tnd_model_validation_20261023/redispatch_probe.py" "$output" "${@:3}"
