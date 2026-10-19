#!/usr/bin/env bash
# 沿用已验证的真实query形状回归，不补成S6，不计性能。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${TASK_DEVICE:?必须经task-submit分配设备}"
export PYPTO_PROG_BUILD_DIR="$here/score_split_query_b1_s1_v1/build_output"
export PYPTO_CACHE_DIR="$here/score_split_query_b1_s1_v1/cache"
exec bash "$here/run_isolated.sh" "$here/score_split_query_regression/query_shape_pair.py" \
  --output "$here/score_split_query_b1_s1_v1" --batch 1 --history 131072 --query-tokens 1 "$@"
