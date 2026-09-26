#!/usr/bin/env bash
# 纯 CPU 解析，不占 NPU；保留每个 rank 的原始 trace 和设备区间。
set -eo pipefail
batch="${1:?指定 4/8/16/24/32/40}"
case "$batch" in 4|8|16|24|32|40) ;; *) exit 2 ;; esac
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
result_root="$repo_root/tests/pypto_test/results/csa_baseline_20260926/model_128k_performance/capacity40/b$batch"
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PYTHONPATH="$repo_root/tests/pypto_test:$PYTHONPATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py profile-export \
        --bank tests/pypto_test/results/release_offline_pd_20260923/h131072_bank \
        --output "$result_root/$backend" --profile-ranks all --analyse-processes 16 \
        > "$result_root/$backend/profile_export.log" 2>&1
done
python tests/pypto_test/offline_pd/performance.py --root "$result_root" \
    --bank tests/pypto_test/results/release_offline_pd_20260923/h131072_bank \
    --mode 2 --batch "$batch" --max-num-seqs 40 --decode-tokens 192 --profile-steps 3
