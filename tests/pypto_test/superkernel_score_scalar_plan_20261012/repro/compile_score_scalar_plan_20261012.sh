#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/tmr_csa_score_scalar_plan_compile.py \
  --reuse "$p/artifacts/csa_scalar_prefix_layout_128k_b16_20261012/compiled" \
  --output "$p/artifacts/csa_score_scalar_plan_128k_b16_20261012/compiled" \
  > "$p/artifacts/csa_score_scalar_plan_compile_20261012.log" 2>&1
