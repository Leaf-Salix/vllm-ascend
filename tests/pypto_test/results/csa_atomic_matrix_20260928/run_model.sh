#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit with 16 devices}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-forward-boundary-71153bb3"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$source_repo:$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_DETERMINISTIC=false
for history in 131072 8192; do
    out="$repo/tests/pypto_test/results/csa_atomic_matrix_20260928/model/h$history"
    bank="$repo/tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank"
    case "$history" in
        131072) batches=(4 8 16); budget=256; port=30081 ;;
        8192) batches=(16 24 32 40); budget=400; port=30091 ;;
    esac
    mkdir -p "$out"
    cd "$out"
    for backend in pto native; do
        mkdir -p "$out/$backend/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/$backend/ascend"
        python "$source_repo/tests/pypto_test/offline_pd/run.py" performance \
            --bank "$bank" --output "$out/$backend" --backend "$backend" --port "$port" \
            --batch 40 --sweep-batches "${batches[@]}" --max-num-batched-tokens "$budget" \
            --decode-tokens 128 --weight-nz-mode 2 --graph-mode full_decode_only \
            --capture-sizes 24 48 96 144 192 240 --warmup-rounds 1 --warmup-tokens 96 \
            --warmup-steps 8 --steady-cycles 10 --profile-start-step 8 --profile-steps 3 \
            > "$out/${backend}_launch.log" 2>&1
    done
done
