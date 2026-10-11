#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto
bash ../artifacts/run_scalar_word_probe_20261011.sh
bash ../artifacts/run_csa_scalar_word_128k_b16_20261011.sh
