#!/usr/bin/env bash
set -eo pipefail
private_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment is required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$private_root/vllm-ascend"
source "$private_root/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/tmr_csa_full_128k_b16_demo.py \
    --device "$TASK_DEVICE" --variants baseline all \
    --compiled "$private_root/artifacts/csa_original_dispatch_compile" \
    --output "$private_root/artifacts/csa_original_dispatch_device"
