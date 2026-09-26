#!/usr/bin/env bash
# 当前源码的整模型正确性基线；固定规约与确定性开关，不作为部署性能结论。
set -eo pipefail
result_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$result_root/../../../../.." && pwd)"
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export ASCEND_PROCESS_LOG_PATH="$result_root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py decode \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h8192_bank \
        --output "$result_root/$backend" --backend "$backend" \
        --batch 16 --decode-tokens 96 --weight-nz-mode 1 \
        --graph-mode full_decode_only --capture-sizes 96 --deterministic
done
