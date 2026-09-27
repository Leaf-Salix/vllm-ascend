#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for case in '131072 8' '131072 16' '8192 24'; do
    read -r history batch <<< "$case"
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v10_unified_followup "$history" "$batch" timing
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v10_unified_followup "$history" "$batch" swimlane
done
