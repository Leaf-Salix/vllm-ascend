#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" baseline_merged 2 6 compare_baseline_t12
bash "$candidate_root/run_case.sh" stage1_selective_v1 2 6 compare_m16_t12
