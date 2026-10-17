#!/usr/bin/env bash
# 获准的唯一修正版CPU；v1失败完整保留，不初始化NPU，不覆盖产物。
set -euo pipefail
qb_evidence_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test ! -e "$qb_evidence_dir/cpu_v2"
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh \
    "$qb_evidence_dir/cpu_v2" --strategy long --attention-query-metadata off \
    --spmd-source "$qb_evidence_dir/source_candidate_v2" \
    --controls-json "$qb_evidence_dir/controls.json" > "$qb_evidence_dir/cpu_v2.log" 2>&1
