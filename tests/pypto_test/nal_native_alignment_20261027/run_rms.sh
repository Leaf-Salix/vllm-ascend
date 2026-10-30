#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
case "$TASK_DEVICE" in
  8|9|10|11|12|13|15) ;;
  *) printf 'This single-card probe requires physical device 8-13 or 15; device 14 is forbidden.\n' >&2; exit 2 ;;
esac
job_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$job_here/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
label=${1:?}
probe=${2:?}
shift 2
[[ "$label" =~ ^[a-z0-9-]+$ && "$probe" =~ ^(rms|hc|spmd|padding|qr|qa240|wkv|kvnorm|qoutput|mainproj|indexproj|indexpool|idxquery|idxquant|weightproj|weight240|attention|woa|woatnd|oproj|compnorm|comprope|topktie|woa240)$ ]]
probe_args=()
while (( $# )); do
  case "$1" in
    --full-batch)
      [[ "$probe" == padding && $# -ge 2 && ${#probe_args[@]} -eq 0 ]]
      [[ "$2" == 1 || "$2" == 40 ]]
      probe_args=(--full-batch "$2")
      ;;
    --query-rows)
      [[ "$probe" == padding && $# -ge 2 && ${#probe_args[@]} -eq 2 && ${probe_args[0]} == --full-batch ]]
      [[ "$2" =~ ^[1-6]$ ]]
      probe_args+=(--query-rows "$2")
      ;;
    --mixed|--projection)
      [[ "$probe" == padding && ${#probe_args[@]} -eq 0 ]]
      probe_args=("$1")
      shift
      continue
      ;;
    --lengths)
      [[ "$probe" == spmd && $# -ge 2 && ${#probe_args[@]} -eq 0 ]]
      [[ "$2" =~ ^[1-6](,[1-6]){0,39}$ ]]
      probe_args=(--lengths "$2")
      ;;
    --layer)
      [[ "$probe" == spmd && $# -ge 2 && ${#probe_args[@]} -eq 2 && ${probe_args[0]} == --lengths ]]
      [[ "$2" =~ ^[0-9]+$ ]]
      (( 2 <= 10#$2 && 10#$2 <= 42 && 10#$2 % 2 == 0 ))
      probe_args+=(--layer "$2")
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
if [[ "$probe" == spmd || "$probe" == padding || "$probe" == topktie ]]; then
  export ASCEND_OPP_PATH
  ASCEND_OPP_PATH=$(python "$SOURCE_ROOT/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/opp-env")
fi
printf 'TASK_RUN_START %s\n' "$(date -Iseconds)"
printf 'TASK_PHYSICAL_DEVICE %s\n' "$TASK_DEVICE"
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
