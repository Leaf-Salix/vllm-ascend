#!/usr/bin/env bash
# 无模型权重的Host热调用复现，同公共SDK；设备仍由task-submit分配。
set -eo pipefail
hca_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hca_repo=$(cd -- "$hca_here/../../.." && pwd)
hca_workspace=$(dirname -- "$hca_repo")
source "$hca_workspace/env-dsv4-0251rc1.sh"
source "$hca_repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec python "$hca_repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$hca_workspace" \
    --vllm-root "$hca_workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$hca_here/repro_eager_host_lookup.py" "$@"
