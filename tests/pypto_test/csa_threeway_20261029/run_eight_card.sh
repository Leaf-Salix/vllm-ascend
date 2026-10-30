#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DEVICE:?task-submit required}"
# 用户本轮临时批准14卡，仅接受最后八卡，禁止使用物理0–7。
[[ "$TASK_DEVICE" == 8,9,10,11,12,13,14,15 ]]
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
align_root=$(cd -- "$here/../../../.." && pwd)
source "$align_root/src/tests/pypto_test/nal_native_alignment_20261027/env.sh"
cohort=${1:?}
arm=${2:?}
label=${3:?}
reference=${4:?use none for Native}
shift 4
[[ "$arm" =~ ^(native|ours)$ && "$label" =~ ^[a-z0-9-]+$ ]]
startup=()
while (( $# )); do
  case "$1" in
    --startup-only)
      [[ "$arm" == native && ${#startup[@]} == 0 ]]
      startup=(--startup-only)
      shift
      ;;
    --device)
      [[ $# -ge 2 && "$2" == "$TASK_DEVICE" ]]
      shift 2
      ;;
    *) exit 2 ;;
  esac
done
snapshot="$cohort/$arm/source"
python - "$snapshot/../manifest.json" <<'PY'
import json
import sys
assert json.load(open(sys.argv[1]))['decode_dp'] == 8
PY
out="$align_root/results/$label"
test ! -e "$out"
mkdir -p "$out"
export SOURCE_ROOT="$snapshot" PTO_CSA_VARIANT=spmd_tnd PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export VLLM_ASCEND_PTO_CSA_BACKEND=python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ASCEND_CACHE_PATH="$out/ascend-cache" ASCEND_PROCESS_LOG_PATH="$out/ascend-log"
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
unset OFFLINE_PTO_SWIMLANE_DIR
export ASCEND_OPP_PATH
ASCEND_OPP_PATH=$(python "$snapshot/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/opp-env")
bank=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/low-acceptance-20261010/banks/fixed8192
args=()
if [[ "$reference" != none ]]; then args=(--reference "$reference"); fi
printf 'TASK_RUN_START %s\n' "$(date -Iseconds)"
printf 'TASK_DEVICE=%s SDK=%s PTOAS_ROOT=%s ASCEND_HOME_PATH=%s\n' \
  "$TASK_DEVICE" "$CSA_SHARED_TOOLCHAIN_RELEASE" "$PTOAS_ROOT" "$ASCEND_HOME_PATH"
trap 'printf "TASK_RUN_END %s\n" "$(date -Iseconds)"' EXIT
python "$here/launch.py" "$snapshot" "$bank" "$out" "${args[@]}" "${startup[@]}"
