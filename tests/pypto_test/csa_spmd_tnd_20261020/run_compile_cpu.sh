#!/usr/bin/env bash
# Compile with CPU meta tensors and the frozen private toolchain; no NPU run.
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
[[ -f "$workspace/env-dsv4-0251rc1.sh" ]]
compile_source=${1:?source package}
compile_descriptors=${2:?descriptor JSON}
compile_output=${3:?fresh output directory}
source "$workspace/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
export PYPTO_PROG_BUILD_DIR="$compile_output/programs" PYPTO_CACHE_DIR="$compile_output/cache"
helpers="$repo/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
export PYTHONPATH="$helpers:$repo/tests/pypto_test:${PYTHONPATH:-}"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    --pypto-root "$workspace/.worktrees/pypto-csa-tnd-perf-20261021" \
    --simpler-root "$workspace/.worktrees/simpler-csa-core-args-256-20261016" \
    --ptoas-root "$workspace/.worktrees/ptoas-csa-tnd-perf-20261021/build" \
    "$here/check_capacity_compile.py" --source "$compile_source" \
    --descriptors "$compile_descriptors" --output "$compile_output"
