#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
: "${TASK_DEVICE:?must use task-submit}"
source "$workspace_dir/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
cd -- "$here/build_output/device_replay_v2"
./replay_device
python "$here/validate_replay.py" --case "$here/build_output/device_replay_v2" --output "$here/DEVICE_REPLAY_VALIDATION.json"
