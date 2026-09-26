#!/usr/bin/env bash
# Performance candidate token guard: fixed weights/bank; both sides use NZ mode 2.
set -eo pipefail
result_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$result_root/../../../../.." && pwd)"
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=1
export HCCL_DETERMINISTIC=false
export ASCEND_PROCESS_LOG_PATH="$result_root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py decode \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
        --output "$result_root/$backend" --backend "$backend" \
        --batch 16 --decode-tokens 96 --weight-nz-mode 2 \
        --graph-mode full_decode_only --capture-sizes 96
done
python tests/pypto_test/offline_pd/compare.py \
    --native "$result_root/native" --pto "$result_root/pto" \
    --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
    --batch 16 --decode-tokens 96 --ranks 16 --output "$result_root/comparison.json"
