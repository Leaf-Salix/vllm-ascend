#!/usr/bin/env bash
set -euo pipefail
E=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qb_qr_b_resident_cpu_20261018
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
test ! -e "$E/cpu_v1"
bash "$D/reproduce_cpu.sh" "$E/cpu_v1" \
  --attention-query-metadata off \
  --spmd-source "$E/source_candidate" \
  --controls-json "$E/controls.json" \
  > "$E/cpu_v1.stdout.log" 2>&1
