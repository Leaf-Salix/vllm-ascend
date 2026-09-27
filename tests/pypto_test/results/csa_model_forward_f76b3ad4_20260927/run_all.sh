#!/usr/bin/env bash
set -eo pipefail
repo=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
result="$repo/tests/pypto_test/results/csa_model_forward_f76b3ad4_20260927"
for history in 8192 131072; do
    bash "$result/run_context.sh" "$history"
done
