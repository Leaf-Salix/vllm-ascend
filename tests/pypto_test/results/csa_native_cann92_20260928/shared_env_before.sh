#!/usr/bin/env bash
# Isolated release migration environment. No main-branch Native operator package.
export PTO_EAGER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export GCC15_ROOT=/data/ci-runner/pypto-toolchain/toolchain/2026.08.5/gcc
export PTOAS_ROOT="$PTO_EAGER_ROOT/.cache/dsv4-toolchain/ptoas-0.66"
export PATH="$PTOAS_ROOT/bin:$GCC15_ROOT/bin:$PATH"
export CC="$GCC15_ROOT/bin/gcc-15"
export CXX="$GCC15_ROOT/bin/g++-15"
export LD_LIBRARY_PATH="$GCC15_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
source "$PTO_EAGER_ROOT/.venv-dsv4-0251rc1/bin/activate"
source "$PTO_EAGER_ROOT/.cache/dsv4-toolchain/atb-package-9.0.0/atb/set_env.sh" --cxx_abi=1
# Native operators built from the same official release as this checkout.
export ASCEND_CUSTOM_OPP_PATH="$PTO_EAGER_ROOT/vllm-ascend-dsv4-pto-0251rc1/.cache/csa/csa-native-ops-install/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
