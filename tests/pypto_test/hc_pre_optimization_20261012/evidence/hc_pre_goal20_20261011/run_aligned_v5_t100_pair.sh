#!/usr/bin/env bash
set -euo pipefail
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh aligned_stage1_rows_v5 20 5 aligned_v5_t100_pair_candidate
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh latest14_formal_20261012 20 5 aligned_v5_t100_pair_control
