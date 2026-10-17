#!/usr/bin/env bash
# 唯一完整符号CPU；复用只读隔离依赖，不创建设备runtime。
set -euo pipefail
bash '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/reproduce_cpu.sh' '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/oa_pair_full_candidate_20261018/cpu_v1' --attention-query-metadata off --spmd-source '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/oa_pair_full_candidate_20261018/source_candidate' --controls-json '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/oa_pair_full_candidate_20261018/controls.json'
