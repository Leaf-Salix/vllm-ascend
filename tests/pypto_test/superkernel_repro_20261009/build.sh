#!/usr/bin/env bash
set -e -o pipefail
repro_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$repro_dir/../../.." && pwd)
source "$repo_dir/../env-dsv4-0251rc1.sh"
output_dir=${1:-"$repo_dir/tests/pypto_test/results/native_superkernel_repro_20261009"}
mkdir -p "$output_dir/build"
cd "$output_dir/build"
bisheng --version > compiler.txt
bisheng -x asc --npu-arch=dav-2201 -O2 -std=c++17 -save-temps \
    -I "$ASCEND_HOME_PATH/include" -L "$ASCEND_HOME_PATH/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" -lascendcl -lascendsk \
    "$repro_dir/native_early_start.asc" -o native_early_start > build.log 2>&1
printf 'Built: %s/build/native_early_start\n' "$output_dir"
