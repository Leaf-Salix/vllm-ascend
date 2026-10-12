#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 28 6 large_wave_reverse_t168
bash "$candidate_root/run_case.sh" adopted_balanced_acc 28 6 large_wave_reverse_control_t168
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 24 6 large_wave_reverse_t144
bash "$candidate_root/run_case.sh" adopted_balanced_acc 24 6 large_wave_reverse_control_t144
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 28 5 large_wave_t140
