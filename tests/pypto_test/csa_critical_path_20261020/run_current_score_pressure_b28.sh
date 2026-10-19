#!/usr/bin/env bash
# 新正式Score的全72角色阶段和READY/ACK诊断，不用于性能表。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$here/run_isolated.sh" "$here/run_score_pressure_probe.py" \
  --output "$here/current_score_pressure_h131072_b28_v1" \
  --batch 28 --history 131072 --launches 1 --wait-backoff --phase-clock \
  --spmd-source "$here/source_after_score_split" "$@"
