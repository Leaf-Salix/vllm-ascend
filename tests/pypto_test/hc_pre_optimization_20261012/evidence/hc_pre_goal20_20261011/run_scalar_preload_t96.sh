#!/usr/bin/env bash
set -eo pipefail
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh adopted_small_stream_acc 16 6 scalar_preload_control_t96
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh scalar_mix_preload_bf16_v1 16 6 scalar_preload_t96
