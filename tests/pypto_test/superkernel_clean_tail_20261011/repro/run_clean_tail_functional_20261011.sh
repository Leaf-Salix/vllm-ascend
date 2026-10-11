#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto
bash ../artifacts/run_clean_tail_20261011.sh
bash ../artifacts/run_clean_tail_mix_20261011.sh
