#!/usr/bin/env bash
# 主Q_B全局NZ N128列任务重分候选：只编译一次完整动态CSA，保留原130根与动态轴。
# 只使用已有隔离依赖，不安装、不触发NPU；已有输出必须保留，拒绝覆盖。
set -euo pipefail
evidence_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pipeline_dir=$(cd -- "$evidence_dir/../.." && pwd)
if [[ -e "$evidence_dir/cpu_v1" || -e "$evidence_dir/cpu_v1.log" ]]; then
    echo 'cpu_v1或日志已存在，本次唯一CPU命令拒绝重复执行。' >&2
    exit 1
fi
bash "$pipeline_dir/reproduce_cpu.sh" "$evidence_dir/cpu_v1" \
    --strategy long --attention-query-metadata off \
    --spmd-source "$evidence_dir/source_candidate" \
    --controls-json "$evidence_dir/controls.json" 2>&1 | tee "$evidence_dir/cpu_v1.log"
