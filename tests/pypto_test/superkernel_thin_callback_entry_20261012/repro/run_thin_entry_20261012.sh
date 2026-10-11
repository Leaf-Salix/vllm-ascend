#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
small="$p/artifacts/thin_entry_probe_20261012"
mkdir -p "$small/ascend"
export ASCEND_PROCESS_LOG_PATH="$small/ascend"
python tests/st/runtime/scheduling/task_boundary_thin_entry_probe.py \
  --work 4 --steps 0 --repeats 0 --device "$TASK_DEVICE" \
  --compiled "$p/artifacts/thin_entry_probe_compiled_20261012" \
  --output "$small" > "$small/run.log" 2>&1
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
output="$p/artifacts/csa_thin_entry_128k_b16_20261012"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/pypto_test/tmr_csa_thin_entry_demo.py \
  --device "$TASK_DEVICE" --variants baseline mixed thin --batch 16 --history 131072 \
  --compiled "$output/compiled" --output "$output" > "$output/run.log" 2>&1
python - "$output/results.json" <<'PYINNER'
import json
import sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ('status', 'summary', 'eager_exact', 'changed_input_exact')}, indent=2))
PYINNER
