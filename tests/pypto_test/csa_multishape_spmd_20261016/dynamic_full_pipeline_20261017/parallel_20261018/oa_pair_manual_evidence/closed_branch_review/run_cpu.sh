#!/usr/bin/env bash
# 关闭候选分支的唯一符号CPU，不改冻结源。
set -euo pipefail
bash '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh' '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/oa_pair_full_candidate_20261018/closed_branch_review/cpu_oa256' --attention-query-metadata off --spmd-source '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/oa_pair_full_candidate_20261018/source_candidate' --controls-json '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/oa_pair_full_candidate_20261018/closed_branch_review/controls_oa256.json'
