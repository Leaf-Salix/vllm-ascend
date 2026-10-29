#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
case "$TASK_DEVICE" in
  8|9|10|11|12|13|14|15) ;;
  *) printf 'This single-card probe requires physical device 8-15.\n' >&2; exit 2 ;;
esac
job_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$job_here/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
label=${1:?}
probe=${2:?}
shift 2
[[ "$label" =~ ^[a-z0-9-]+$ && "$probe" =~ ^(rms|hc|spmd|qr|wkv|kvnorm|qoutput|mainproj|indexproj|idxquery|idxquant|weightproj|attention|woa|oproj|compnorm)$ ]]
probe_args=()
while (( $# )); do
  case "$1" in
    --lengths)
      [[ "$probe" == spmd && $# -ge 2 && ${#probe_args[@]} -eq 0 ]]
      [[ "$2" =~ ^[1-6](,[1-6]){0,39}$ ]]
      probe_args=(--lengths "$2")
      ;;
    --device) [[ $# -ge 2 && "$2" == "$TASK_DEVICE" ]] ;;
    *) exit 2 ;;
  esac
  shift 2
done
out="$align_root/results/$label"
test ! -e "$out"
mkdir -p "$out"
export ASCEND_CACHE_PATH="$out/ascend-cache" ASCEND_PROCESS_LOG_PATH="$out/ascend-log"
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
if [[ "$probe" == spmd ]]; then
  export ASCEND_OPP_PATH
  ASCEND_OPP_PATH=$(python "$SOURCE_ROOT/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/opp-env")
fi
printf 'TASK_RUN_START %s\n' "$(date -Iseconds)"
trap 'printf "TASK_RUN_END %s\n" "$(date -Iseconds)"' EXIT
phase_args=()
if [[ -n ${HC_PROBE_PHASE:-} ]]; then
  [[ $HC_PROBE_PHASE == after-fold ]]
  phase_args+=("$HC_PROBE_PHASE")
fi
timeout --signal=TERM --kill-after=15s 750s python \
  "$align_root/src/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  "$job_here/probe_$probe.py" "$out" "${phase_args[@]}" "${probe_args[@]}" > "$out/run.log" 2>&1
