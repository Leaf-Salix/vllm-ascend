#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment required}"
output="$p/artifacts/csa_ready_gate_128k_b16_20261012"
mkdir -p "$output/functional/ascend"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
export ASCEND_PROCESS_LOG_PATH="$output/functional/ascend"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_remote_dependency.py \
  tests/st/runtime/scheduling/test_local_task_boundary_mix.py -q -n 1 \
  --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$output/functional/kernels" \
  --junitxml "$output/functional/results.xml" > "$output/functional/run.log" 2>&1
tail -n 3 "$output/functional/run.log"
cd "$p/vllm-ascend"
source "$p/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
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
mkdir -p "$output/coverage/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/coverage/ascend"
python tests/pypto_test/tmr_csa_ready_gate_coverage.py --compiled "$output/compiled" \
  --output "$output/coverage" --device "$TASK_DEVICE" > "$output/coverage/run.log" 2>&1
tail -n 3 "$output/coverage/run.log"
