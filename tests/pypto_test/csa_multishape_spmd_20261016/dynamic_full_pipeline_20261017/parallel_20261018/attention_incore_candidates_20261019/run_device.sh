#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
: "${TASK_DEVICE:?必须由task-submit分配NPU}"
cd -- "$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
source .claude/skills/testing/load-env.sh
cd -- "$repo_dir"
exec bash "$here/../attention_local_stats_policy_evidence/run_device.sh" "$@"
