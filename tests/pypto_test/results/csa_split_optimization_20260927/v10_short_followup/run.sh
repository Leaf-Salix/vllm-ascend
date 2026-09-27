#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for batch in 16 32 40; do
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v10_short_followup 8192 "$batch" timing
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v10_short_followup 8192 "$batch" swimlane
done
