#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v8_native_pair 131072 4 timing
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh v8_native_pair 131072 8 timing
