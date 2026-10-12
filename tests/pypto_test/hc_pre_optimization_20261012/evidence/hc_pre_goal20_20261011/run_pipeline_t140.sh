#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_balanced_acc 28 5 large_pipe_control_t140
bash "$candidate_root/run_case.sh" large_pipeline_selective_v1 28 5 large_pipe_t140
