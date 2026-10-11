#!/usr/bin/env bash
set -eo pipefail
private_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment is required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$private_root/vllm-ascend"
source "$private_root/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
output="$private_root/artifacts/csa_seven_handoff_cleanup_20261011"
for shape in 131072:4 131072:8 131072:16 131072:24 8192:16 8192:24 8192:32; do
    history="${shape%:*}"
    batch="${shape#*:}"
    case_name="h${history}_b${batch}"
    printf 'STAGE: begin %s on device %s\n' "$case_name" "$TASK_DEVICE"
    python tests/pypto_test/tmr_csa_full_128k_b16_demo.py \
        --device "$TASK_DEVICE" --variants baseline all \
        --batch "$batch" --history "$history" \
        --compiled "$output/compiled" --output "$output/$case_name" \
        > "$output/$case_name.log" 2>&1
    python - "$output/$case_name/results.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
print(json.dumps({k: data[k] for k in ("batch", "history", "status", "summary")}), flush=True)
PY
done
