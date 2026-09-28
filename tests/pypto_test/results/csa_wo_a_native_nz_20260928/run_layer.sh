#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_wo_a_native_nz_20260928"
exec bash "$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928/run_layer.sh" \
    "$root" "$workspace/.cache/csa-wo-a-native-nz-3b27c7fd" 16 \
    "$workspace/.cache/csa-cann92-baseline-e33d842a"
