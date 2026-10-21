#!/usr/bin/env bash
# 描述符/路由合同的CPU测试；不分配NPU。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
[[ -f "$workspace/env-dsv4-0251rc1.sh" ]]
source "$workspace/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
cd -- "$repo"
python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    --pypto-root "$workspace/.worktrees/pypto-csa-tnd-perf-20261021" \
    --simpler-root "$workspace/.worktrees/simpler-csa-core-args-256-20261016" \
    --ptoas-root "$workspace/.worktrees/ptoas-csa-tnd-perf-20261021/build" \
    "$here/run_cpu.py" "$@"
