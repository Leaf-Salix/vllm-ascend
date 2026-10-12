#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_balanced_acc 20 5 large_pipe_low_control_t100
bash "$candidate_root/run_case.sh" stage1_large_wave_pipeline_v1 20 5 large_pipe_low_t100
bash "$candidate_root/run_case.sh" adopted_balanced_acc 20 6 large_pipe_low_control_t120
bash "$candidate_root/run_case.sh" stage1_large_wave_pipeline_v1 20 6 large_pipe_low_b20s6
bash "$candidate_root/run_case.sh" stage1_large_wave_pipeline_v1 24 5 large_pipe_low_b24s5
