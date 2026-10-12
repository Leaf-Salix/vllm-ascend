#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_hf32_case.sh" fp32 16 6 mode_fp32_t96
bash "$candidate_root/run_hf32_case.sh" hf32 16 6 mode_hf32_t96
