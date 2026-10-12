#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" large_scalar_mix_v1 28 6 large_scalar_reverse_t168
bash "$candidate_root/run_case.sh" adopted_large_pipeline 28 6 large_scalar_reverse_control_t168
bash "$candidate_root/run_case.sh" adopted_large_pipeline 24 6 large_scalar_control_t144
bash "$candidate_root/run_case.sh" large_scalar_mix_v1 24 6 large_scalar_t144
