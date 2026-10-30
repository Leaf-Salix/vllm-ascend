#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
case "$TASK_DEVICE" in 8|9|10|11|12|13|15) ;; *) exit 2 ;; esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
align_root=$(cd -- "$here/../../../.." && pwd)
source "$align_root/src/tests/pypto_test/nal_native_alignment_20261027/env.sh"
label=${1:?}
batch=${2:?}
shift 2
[[ "$label" =~ ^[a-z0-9-]+$ && ( "$batch" == 1 || "$batch" == 8 || "$batch" == 40 ) ]]
active=()
while (( $# )); do
  case "$1" in
    --active-requests)
      [[ "$batch" == 40 && $# -ge 2 && "$2" =~ ^([1-9]|[1-3][0-9])$ && ${#active[@]} == 0 ]]
      active=(--active-requests "$2")
      shift 2
      ;;
    --device)
      [[ $# -ge 2 && "$2" == "$TASK_DEVICE" ]]
      shift 2
      ;;
    *) exit 2 ;;
  esac
done
out="$align_root/results/$label"
test ! -e "$out"
mkdir -p "$out"
export SOURCE_ROOT="$align_root/cohorts/eightcard-source-v5/native/source"
export PADDING_STAGE_REFERENCE="$align_root/results/t240-layer2-weight-order-native-v1/result.json"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE" PTO_CSA_RING_TASK_WINDOW=2048
export ASCEND_CACHE_PATH="$out/ascend-cache" ASCEND_PROCESS_LOG_PATH="$out/ascend-log"
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
export ASCEND_OPP_PATH
ASCEND_OPP_PATH=$(python "$SOURCE_ROOT/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/opp-env")
args=()
if [[ "$batch" == 8 ]]; then args=(--query-rows 1); fi
timeout --signal=TERM --kill-after=15s 750s python \
  "$align_root/src/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  "$align_root/src/tests/pypto_test/nal_native_alignment_20261027/probe_padding.py" \
  "$out" --full-batch "$batch" "${args[@]}" "${active[@]}" > "$out/run.log" 2>&1
