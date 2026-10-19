#!/usr/bin/env bash
# 诊断driver归档的可执行入口；实际环境与源码选择统一由父目录包装器配置。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$here/../run_failure_probe.sh" "$@"
