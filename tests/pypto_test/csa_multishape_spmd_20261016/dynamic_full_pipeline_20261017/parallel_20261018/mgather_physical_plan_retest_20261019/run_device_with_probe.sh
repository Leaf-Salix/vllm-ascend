#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:?指定全新输出目录}
shift
bash "$here/run_probe.sh" "${out}_mgather_probe"
bash "$here/run_device.sh" "$out" "$@"
