#!/usr/bin/env bash
set -eo pipefail
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh adopted_small_sink 16 5 compact_mid_control_t80
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh sink_compact_block_v3 16 5 compact_mid_t80
