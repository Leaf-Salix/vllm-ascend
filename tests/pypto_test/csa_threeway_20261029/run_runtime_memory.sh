#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
case "$TASK_DEVICE" in 8|9|10|11|12|13|15) ;; *) exit 2 ;; esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
align_root=$(cd -- "$here/../../../.." && pwd)
source "$align_root/src/tests/pypto_test/nal_native_alignment_20261027/env.sh"
label=${1:?}
window=${2:?use default or a positive integer}
pool=${3:?use default or a positive integer}
shift 3
[[ "$label" =~ ^[a-z0-9-]+$ ]]
args=()
for value in "$window" "$pool"; do [[ "$value" == default || "$value" =~ ^[1-9][0-9]*$ ]]; done
if [[ "$window" != default ]]; then args+=(--task-window "$window"); fi
if [[ "$pool" != default ]]; then args+=(--dep-pool "$pool"); fi
if (( $# )); then [[ $# == 2 && "$1" == --device && "$2" == "$TASK_DEVICE" ]]; fi
out="$align_root/results/$label"
test ! -e "$out"
mkdir -p "$out"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_CACHE_PATH="$out/ascend-cache" ASCEND_PROCESS_LOG_PATH="$out/ascend-log"
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
timeout --signal=TERM --kill-after=10s 210s python \
  "$align_root/src/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --native-plugin-root "$align_root/cohorts/eightcard-source-v5/ours/source" \
  "$here/runtime_memory.py" "$out/result.json" "${args[@]}" > "$out/run.log" 2>&1
