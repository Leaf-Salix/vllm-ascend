#!/usr/bin/env bash
# 2026-10-22：后续只测 Native eager 直接调用，不捕获或重放 ACLGraph。
set -eo pipefail
hca_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hca_case=${1:?固定档位名}
hca_output=$(realpath -m -- "${2:?新的输出目录}")
shift 2
read -r hca_history hca_batch hca_queries < <(python - "$hca_here/cases.json" "$hca_case" <<'SPEC'
import json
import sys
case = next(c for c in json.load(open(sys.argv[1]))["cases"] if c["name"] == sys.argv[2])
print(case["history_label"], case["active_requests"], ",".join(map(str, case["query_lengths"])))
SPEC
)
exec bash "$hca_here/launch.sh" --native-only --windows 10 --output "$hca_output" \
    --history "$hca_history" --batch "$hca_batch" --queries "$hca_queries" "$@"
