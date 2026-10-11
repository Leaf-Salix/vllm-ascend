#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
output="$p/artifacts/csa_score_fair_128k_b16_20261012"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/pypto_test/tmr_csa_score_fair_demo.py --variants plain_score early_score \
  --device "$TASK_DEVICE" --batch 16 --history 131072 --compiled "$output/compiled" --output "$output" \
  > "$output/run.log" 2>&1
python - "$output/results.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(json.dumps({k:r[k] for k in ('status','summary','eager_exact','changed_input_exact')},indent=2))
PY
