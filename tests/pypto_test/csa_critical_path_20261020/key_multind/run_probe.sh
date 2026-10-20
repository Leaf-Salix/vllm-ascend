#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${TASK_DEVICE:?通过task-submit --device auto运行}"
exec bash "$here/run_isolated.sh" "$here/probe.py" --output "$here/device_probe_v1" "$@"
