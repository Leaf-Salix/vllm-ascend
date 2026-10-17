#!/usr/bin/env bash
# 同一个队列配卡生命周期：私有plan地址oracle通过后，才运行完整CSA三方对照。
set -eo pipefail
: "${TASK_DEVICE:?必须通过 task-submit --device auto}"
case_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
worktree=$(cd -- "$case_root/../../.." && pwd)
workspace=/data/pyptouser/qinchuanyu/pto-eager
canonical="$workspace/vllm-ascend-dsv4-pto-0251rc1"
experiment="$canonical/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export PYPTO_PROG_BUILD_DIR="$case_root/device_probe_build_v1" PYPTO_CACHE_DIR="$case_root/device_probe_cache_v1"
unset HC_PRE_TEST_OPERATOR_SOURCE
python "$canonical/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$case_root/device_plan_probe.py" "$case_root/device_probe_v1"
bash "$experiment/run_device.sh" "$experiment/parallel_20261018/physical_plan_h131072_b16_v2" \
  --history 131072 --batch 16 \
  --spmd-source "$worktree/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd" \
  --policy-json "$case_root/policy_on.json" \
  --source-matrix-reference "$experiment/parallel_20261018/baseline_source" \
  --wait-backoff --metadata-replay --l1-windows 5
