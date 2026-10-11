#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto
bash ../artifacts/run_callback_recheck_probe_20261011.sh
bash ../artifacts/run_csa_callback_recheck_128k_b16_20261011.sh
