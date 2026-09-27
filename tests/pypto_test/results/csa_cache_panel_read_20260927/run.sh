#!/usr/bin/env bash
set -eo pipefail
repo=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for history in 8192 131072; do
  for kind in timing swimlane; do
    bash "$repo/tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh" cache_panel_read "$history" 16 "$kind"
  done
done
