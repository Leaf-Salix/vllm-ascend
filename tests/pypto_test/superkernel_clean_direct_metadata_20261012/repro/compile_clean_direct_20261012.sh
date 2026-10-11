#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
python tests/st/runtime/scheduling/task_boundary_clean_direct_probe.py \
  --compile --work 4 --steps 0 --compiled "$p/artifacts/clean_direct_probe_compiled_20261012" \
  > "$p/artifacts/clean_direct_probe_compile_20261012.log" 2>&1
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/tmr_csa_clean_direct_compile.py \
  --output "$p/artifacts/csa_clean_direct_128k_b16_20261012/compiled" \
  > "$p/artifacts/csa_clean_direct_compile_20261012.log" 2>&1
