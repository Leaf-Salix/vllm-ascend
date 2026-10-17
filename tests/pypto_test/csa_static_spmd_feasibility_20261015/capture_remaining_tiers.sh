#!/usr/bin/env bash
# B4/H131072 已完成；其余六档各补一个映射窗口，不重复性能测试。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配卡}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$(dirname "$here")/results/csa_static_spmd_feasibility_20261015"
for tier in 131072:8 131072:16 131072:24 8192:16 8192:24 8192:32; do
    history="${tier%:*}"
    batch="${tier#*:}"
    bash "$here/capture_dispatch.sh" "$out/mapping_h${history}_b${batch}_v1" "$history" "$batch"
done
