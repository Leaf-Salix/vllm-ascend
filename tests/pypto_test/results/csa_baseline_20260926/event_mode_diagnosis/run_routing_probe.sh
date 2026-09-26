#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配 1 卡}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source ../env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
python tests/pypto_test/offline_pd/moe_routing.py --device 0
