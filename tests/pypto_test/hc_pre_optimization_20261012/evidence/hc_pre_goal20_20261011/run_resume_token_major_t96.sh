#!/usr/bin/env bash
set -euo pipefail
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh adopted_l0_interleave_exact96 16 6 resume_token_major_t96_control
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh token_major_projection_v2 16 6 resume_token_major_t96_candidate
