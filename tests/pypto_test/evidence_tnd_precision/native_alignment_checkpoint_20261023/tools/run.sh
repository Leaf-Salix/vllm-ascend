#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
stage_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
label=${1:?}
[[ "$label" =~ ^st-[a-z0-9-]+-v[0-9]+$ ]]
base=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/threeway-small-20261022-codex/capture-v2
source "$base/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
out="$stage_root/results/$label"
test ! -e "$out"
mkdir -p "$stage_root/results"
export PREFLIGHT_RECEIPT="$out-preflight.json"
export ASCEND_CACHE_PATH="$out-ascend-cache" ASCEND_PROCESS_LOG_PATH="$out-ascend-log" VLLM_CACHE_ROOT="$out-vllm-cache"
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
export ASCEND_OPP_PATH
ASCEND_OPP_PATH=$(python "$SOURCE_ROOT/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out-opp-env")
lengths=6
for ((i=1;i<32;i++)); do lengths+=,6; done
ccache -s > "$out-ccache.txt"
timeout --signal=TERM --kill-after=15s 750s python "$stage_root/launch.py" --out "$out" --lengths "$lengths" --layer 4 --history 8192 --mode timing > "$out-run.log" 2>&1
