#!/usr/bin/env bash
set -eo pipefail
runner="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_case.sh"
history="${1:?history}"
batch="${2:?batch}"
bash "$runner" v4 "$history" "$batch" timing
bash "$runner" v7 "$history" "$batch" timing
bash "$runner" v7 "$history" "$batch" swimlane
