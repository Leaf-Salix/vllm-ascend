#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single queue-assigned card required}"
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PYPTO_CODEGEN_MAX_WORKERS="$PYPTO_BUILD_JOBS"
output="$p/artifacts/csa_late_coverage_128k_b16_20261011"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/pypto_test/tmr_csa_prefix_coverage.py --device "$TASK_DEVICE" \
  --compiled "$p/artifacts/csa_event_contract_128k_b16_20261011/compiled" \
  --output "$output" > "$output/run.log" 2>&1
python - "$output/results.json" <<'PYINNER'
import json
import sys
r = json.load(open(sys.argv[1]))
print(r['status'])
print(json.dumps([s['totals'] for s in r['samples']], indent=2))
PYINNER
