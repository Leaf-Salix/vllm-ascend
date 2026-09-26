#!/usr/bin/env bash
# 单卡正式层权重对照；设备由 task-submit 分配。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 提交单卡任务}"
if [[ "$TASK_DEVICE" == *,* ]]; then
  printf '整层诊断只使用一张卡。\n' >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "$repo_root")/env-dsv4-0251rc1.sh"
case_output="${1:?指定新的结果目录}"
shift
mkdir -p "$case_output"
case_output="$(realpath "$case_output")"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_PROCESS_LOG_PATH="$case_output/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd "$case_output"
exec python "$repo_root/tests/pypto_test/dsv4_csa_single_layer.py" \
  --device 0 --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
  --output "$case_output" "$@"
