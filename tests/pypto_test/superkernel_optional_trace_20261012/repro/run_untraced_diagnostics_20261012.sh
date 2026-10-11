#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
output="$p/artifacts/untraced_notice_diagnostics_20261012"
mkdir -p "$output/device/ascend"
export ASCEND_PROCESS_LOG_PATH="$output/device/ascend"
python tests/st/runtime/scheduling/task_boundary_runtime_poll_probe.py \
 --compiled "$output/compiled" --output "$output/device" --device "$TASK_DEVICE" \
 --repeats 0 --work 1 > "$output/device/run.log" 2>&1
python - "$output/device/results.json" <<'INNER'
import json, sys
r = json.load(open(sys.argv[1]))
for case in r['cases']:
    counts = [len(x['polled_after_first_miss']) for x in case['rounds'] if x['mode'] == 'late']
    assert counts and sum(counts) > 0, (case['cube'], counts)
    print({'cube': case['cube'], 'notification_counts': counts, 'functional': 'PASS'})
INNER
