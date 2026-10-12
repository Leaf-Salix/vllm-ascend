#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/csa_gated_args_128k_b16_20261012"
mkdir -p "$output/diagnostics/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/diagnostics/ascend"
export PYPTO_CODEGEN_MAX_WORKERS="$PYPTO_BUILD_JOBS"
PYTHONPATH="$p/pypto:$PYTHONPATH" python - "$output" <<'INNER' > "$output/diagnostics/run.log" 2>&1
import os, sys
from pathlib import Path
from simpler_setup.runtime_builder import RuntimeBuilder
output = Path(sys.argv[1])
RuntimeBuilder._CACHE_DIR = output / 'runtime/cache'
RuntimeBuilder._LIB_DIR = output / 'runtime/lib'
import pytest
raise SystemExit(pytest.main([
    'tests/st/runtime/scheduling/test_local_task_boundary_overlap.py',
    'tests/st/runtime/scheduling/test_local_task_boundary_mix.py',
    '-k', '(local_boundary_overlap and flat_wait) or (mixed_members_enter_prefix and default_events)',
    '--platform', 'a2a3', '--device', os.environ['TASK_DEVICE'],
    '--save-kernels', '--kernels-dir', str(output/'diagnostics/kernels'),
    '-q', '--junitxml', str(output/'diagnostics/results.xml'),
]))
INNER
tail -3 "$output/diagnostics/run.log"
cd "$p/vllm-ascend"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
for mode in control_before candidate control_after; do
    mkdir -p "$output/$mode/ascend"
    export ASCEND_PROCESS_LOG_PATH="$output/$mode/ascend"
    if [ "$mode" = candidate ]; then
        python tests/pypto_test/tmr_csa_gated_args_demo.py \
            --runtime-root "$output/runtime" \
            --variants plain_score early_score --device "$TASK_DEVICE" \
            --batch 16 --history 131072 \
            --compiled "$p/artifacts/csa_consumed_context_128k_b16_20261012/retained/compiled" \
            --output "$output/$mode" > "$output/$mode/run.log" 2>&1
    else
        python tests/pypto_test/tmr_csa_score_fair_demo.py \
            --variants plain_score early_score --device "$TASK_DEVICE" \
            --batch 16 --history 131072 \
            --compiled "$p/artifacts/csa_consumed_context_128k_b16_20261012/retained/compiled" \
            --output "$output/$mode" > "$output/$mode/run.log" 2>&1
    fi
    python - "$output/$mode/results.json" <<'INNER'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({'source': sys.argv[1], 'status': r['status'], 'summary': r['summary']}, indent=2))
INNER
done
