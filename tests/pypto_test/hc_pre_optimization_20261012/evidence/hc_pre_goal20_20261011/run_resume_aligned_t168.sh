#!/usr/bin/env bash
set -euo pipefail
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh adopted_l0_interleave_exact96 28 6 resume_aligned_t168_control
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh aligned_stage1_rows_v4 28 6 resume_aligned_t168_candidate
