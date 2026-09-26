#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source ../env-dsv4-0251rc1.sh
root="$PWD/tests/pypto_test/results/csa_baseline_20260926/event_mode_diagnosis"
export ASCEND_PROCESS_LOG_PATH="$root/ascend_probe"
for mode in default 0; do
    options=()
    if [[ "$mode" != default ]]; then options=(--mode "$mode"); fi
    python tests/pypto_test/offline_pd/event_mode.py --device "$TASK_DEVICE" \
        --output "$root/probe_${mode}.json" "${options[@]}"
done
