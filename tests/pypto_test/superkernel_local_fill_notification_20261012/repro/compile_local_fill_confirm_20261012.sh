#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/tmr_csa_local_fill_compile.py \
  --reuse "$p/artifacts/csa_clean_direct_128k_b16_20261012/compiled" \
  --output "$p/artifacts/csa_local_fill_confirm_128k_b16_20261012/compiled" \
  > "$p/artifacts/csa_local_fill_confirm_compile_20261012.log" 2>&1
