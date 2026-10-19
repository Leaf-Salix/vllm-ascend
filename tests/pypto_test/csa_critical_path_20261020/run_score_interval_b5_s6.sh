#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${TASK_DEVICE:?必须经队列分配设备}"
export PYPTO_PROG_BUILD_DIR="$here/score_interval_query_b5_s6_v1/build_output"
export PYPTO_CACHE_DIR="$here/score_interval_query_b5_s6_v1/cache"
exec bash "$here/run_isolated.sh" "$here/score_interval_query_regression/query_shape_pair.py" \
  --output "$here/score_interval_query_b5_s6_v1" --batch 5 --history 131072 --query-tokens 6 "$@"
