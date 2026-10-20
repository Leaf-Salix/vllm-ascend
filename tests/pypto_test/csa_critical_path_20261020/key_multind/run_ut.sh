#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
cd -- "$workspace/.worktrees/pypto-csa-key-multind-20261020"
source .claude/skills/testing/load-env.sh
export CSA_MULTIND_EVIDENCE="$here"
exec bash "$here/run_isolated.sh" "$here/run_ut.py"
