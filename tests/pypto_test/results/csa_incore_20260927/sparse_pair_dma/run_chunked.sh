#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_incore_20260927/sparse_pair_dma"
mkdir -p "$root/chunked/ascend"
export ASCEND_PROCESS_LOG_PATH="$root/chunked/ascend"
cd "$root"
python run.py --chunked > chunked/run.log 2>&1
