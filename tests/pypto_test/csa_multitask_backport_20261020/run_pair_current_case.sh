#!/usr/bin/env bash
# 新阶段显式采用7061c112的150tick参考；旧run_pair_case入口仍可复现100tick历史。
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$here/run_pair_case.sh" --spmd-reference spmd_7061c112 "$@"
