#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/vllm-ascend"
source "$p/env.sh"
output="$p/artifacts/csa_dispatch_coverage_128k_b16_20261012"
mkdir -p "$output/ascend" "$output/controls"
if [ ! -e "$output/controls/baseline" ]; then
    ln -s "$p/artifacts/csa_untraced_notice_128k_b16_20261012/compiled/plain_score" "$output/controls/baseline"
fi
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PYPTO_CODEGEN_MAX_WORKERS="$PYPTO_BUILD_JOBS"
python tests/pypto_test/tmr_csa_dispatch_coverage.py \
    --device "$TASK_DEVICE" --compiled "$output/controls" --output "$output" \
    > "$output/run.log" 2>&1
python - "$output/results.json" <<'INNER'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({'status': r['status'], 'rounds': [x['publication']['counts'] for x in r['samples']]}, indent=2))
INNER
