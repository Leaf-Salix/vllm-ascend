#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
small="$p/artifacts/local_fill_probe_20261012"
mkdir -p "$small/ascend"
export ASCEND_PROCESS_LOG_PATH="$small/ascend"
python tests/st/runtime/scheduling/task_boundary_local_fill_probe.py \
  --work 4 --steps 0 --repeats 0 --device "$TASK_DEVICE" \
  --compiled "$p/artifacts/local_fill_probe_compiled_20261012" \
  --output "$small" > "$small/run.log" 2>&1
mixed="$p/artifacts/local_fill_mix_20261012"
mkdir -p "$mixed/ascend"
export ASCEND_PROCESS_LOG_PATH="$mixed/ascend"
python - "$TASK_DEVICE" "$mixed" > "$mixed/run.log" 2>&1 <<'PYINNER'
import sys
import pytest
sys.path.insert(0, 'tests/st/runtime/scheduling')
from task_boundary_local_fill_probe import local_fill_notification
with local_fill_notification():
    # This process owns the wrapper hook; run its two integration checks in
    # process, serially, on the one explicitly assigned device.
    code = pytest.main(['tests/st/runtime/scheduling/test_local_task_boundary_mix.py',
        '-q', '-n', '0', '--platform', 'a2a3', '--device', sys.argv[1], '--save-kernels',
        '--kernels-dir', sys.argv[2], '--junitxml', sys.argv[2] + '/junit.xml'])
raise SystemExit(code)
PYINNER
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
output="$p/artifacts/csa_local_fill_128k_b16_20261012"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/pypto_test/tmr_csa_local_fill_demo.py \
  --device "$TASK_DEVICE" --variants baseline mixed local_fill --batch 16 --history 131072 \
  --compiled "$output/compiled" --output "$output" > "$output/run.log" 2>&1
python - "$output/results.json" <<'PYINNER'
import json
import sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ('status', 'summary', 'eager_exact', 'changed_input_exact')}, indent=2))
PYINNER
