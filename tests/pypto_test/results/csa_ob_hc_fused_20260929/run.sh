#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit this paired experiment through task-submit --device auto}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Baseline includes validated O-B reuse; only final AIV task organization changes.
test -f "$root/compile_candidate.json"
test -f "$root/compile_shared_hc.json"
test -f "$root/static_evidence.json"
for case_spec in 131072:16 8192:24; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    sides=(baseline candidate)
    if [[ "$history" == 8192 ]]; then sides=(candidate baseline); fi
    for phase in timing swimlane; do
        for side in "${sides[@]}"; do
            bash "$root/run_side.sh" "$side" "$phase" "$history" "$batch"
        done
    done
done
