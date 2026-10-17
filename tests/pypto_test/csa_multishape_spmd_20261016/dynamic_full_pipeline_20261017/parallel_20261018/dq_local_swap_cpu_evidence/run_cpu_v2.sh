#!/usr/bin/env bash
# 唯一获准的INT16修正版CPU生成；v1源码、日志和失败产物不覆盖。
set -euo pipefail
candidate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
probe_dir=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017
output_dir=${1:-$candidate_dir/cpu_v2}
test ! -e "$output_dir"
bash "$probe_dir/reproduce_cpu.sh" "$output_dir" --spmd-source "$candidate_dir/source_v2" --controls-json "$candidate_dir/controls.json"
