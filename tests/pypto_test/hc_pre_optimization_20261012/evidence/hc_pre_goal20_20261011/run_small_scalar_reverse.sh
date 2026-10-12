#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" small_scalar_partition_v2 2 6 small_scalar_reverse_t12
bash "$candidate_root/run_case.sh" adopted_large_pipeline 2 6 small_scalar_reverse_control_t12
bash "$candidate_root/run_case.sh" small_scalar_partition_v2 8 6 small_scalar_reverse_t48
bash "$candidate_root/run_case.sh" adopted_large_pipeline 8 6 small_scalar_reverse_control_t48
bash "$candidate_root/run_case.sh" small_scalar_partition_v2 2 5 small_scalar_t10
