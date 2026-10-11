#!/usr/bin/env bash
set -eo pipefail
normal_notice_suffix=""
if [ "${1:-}" = "confirm" ]; then
    normal_notice_suffix="_confirm"
fi
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/vllm-ascend"
source "$p/env.sh"
output="$p/artifacts/csa_normal_notice_128k_b16_20261012$normal_notice_suffix"
mkdir -p "$output/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/ascend"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python tests/pypto_test/tmr_csa_normal_notice_demo.py \
 --variants plain_score counted_notice normal_notice \
 --device "$TASK_DEVICE" --batch 16 --history 131072 \
 --compiled "$p/artifacts/csa_normal_notice_128k_b16_20261012/compiled" --output "$output" \
 > "$output/run.log" 2>&1
python - "$output/results.json" <<'INNER'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ('status', 'summary')}, indent=2))
INNER
