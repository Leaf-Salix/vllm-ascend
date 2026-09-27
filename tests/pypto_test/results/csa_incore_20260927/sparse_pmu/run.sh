#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_pmu"
case_dir="$root/native"
mkdir -p "$case_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
cd "$case_dir"
python "$repo/tests/pypto_test/dsv4_csa_single_layer.py" \
    --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
    --output "$case_dir" --device "$TASK_DEVICE" --batch 40 --history 8192 \
    --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
    --deterministic-level 0 --native-profile-only --save-sparse-case \
    --timing-iters 1 --timing-warmup 5 > "$case_dir/run.log" 2>&1
case_dir="$root/standalone_pipe"
mkdir -p "$case_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
cd "$case_dir"
exec python "$repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" \
    --input "$root/native/native_sparse.pt" --output "$case_dir" \
    --variant performance --pmu 2 > "$case_dir/run.log" 2>&1
