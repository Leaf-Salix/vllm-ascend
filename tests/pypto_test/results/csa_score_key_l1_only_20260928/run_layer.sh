#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
exec bash "$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928/run_layer.sh" \
    "$repo/tests/pypto_test/results/csa_score_key_l1_only_20260928" \
    "$workspace/.cache/csa-score-key-l1-only-554b3bca" 8
