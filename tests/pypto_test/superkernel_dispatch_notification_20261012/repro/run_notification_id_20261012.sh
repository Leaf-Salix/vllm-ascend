#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/csa_notification_id_128k_b16_20261012"
mkdir -p "$output/ascend" "$output/functional"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
python tests/st/runtime/scheduling/task_boundary_runtime_poll_probe.py \
 --compiled "$p/artifacts/notification_id_probe_20261012/compiled" --output "$output/functional/generation" \
 --device "$TASK_DEVICE" --repeats 0 --work 1 > "$output/functional/generation.log" 2>&1
python tests/st/runtime/scheduling/task_boundary_notification_wrap_probe.py \
 --compiled "$p/artifacts/notification_id_wrap_probe_20261012/compiled" --output "$output/functional/wrapped" \
 --device "$TASK_DEVICE" --repeats 0 --work 1 > "$output/functional/wrapped.log" 2>&1
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_mix.py -q \
 -k 'mixed_local_boundaries or default_events' --platform a2a3 --device "$TASK_DEVICE" \
 --save-kernels --kernels-dir "$output/functional/mix" --junitxml "$output/functional/mix.xml" \
 > "$output/functional/mix.log" 2>&1
cd "$p/vllm-ascend"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/tmr_csa_notification_id_demo.py --variants plain_score original_notice dispatch_notice \
 --device "$TASK_DEVICE" --batch 16 --history 131072 --compiled "$output/compiled" --output "$output" \
 > "$output/run.log" 2>&1
python - "$output/results.json" <<'INNER'
import json,sys
r=json.load(open(sys.argv[1]))
print(json.dumps({k:r[k] for k in ('status','summary')},indent=2))
INNER
