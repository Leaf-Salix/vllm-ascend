#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
source "$p/env.sh"
cd "$p/vllm-ascend"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
output="$p/artifacts/csa_compiled_poll_128k_b16_20261011"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/pypto_test/tmr_csa_late_poll_demo.py \
  --device "$TASK_DEVICE" --variants baseline once mixed --batch 16 --history 131072 \
  --compiled "$p/artifacts/csa_compiled_poll_128k_b16_20261011/compiled" \
  --output "$output" > "$output/run.log" 2>&1
python - "$output/results.json" <<'PYINNER'
import json
import sys
r=json.load(open(sys.argv[1]))
print(json.dumps({k:r[k] for k in ('status','summary','eager_exact','changed_input_exact')},indent=2))
PYINNER
