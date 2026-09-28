#!/usr/bin/env bash
# Source in an isolated subprocess; keep Python, ATB and custom kernels fixed.
export PTO_EAGER_ROOT=/data/pyptouser/qinchuanyu/pto-eager
if declare -F deactivate >/dev/null; then
    deactivate nondestructive
fi
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH ASCEND_CUSTOM_OPP_PATH
unset ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME
case "${1:?choose cann90 or cann92}" in
    cann90) source /usr/local/Ascend/cann-9.0.0/set_env.sh ;;
    cann92) source /data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2/set_env.sh ;;
    *) return 2 ;;
esac
export GCC15_ROOT=/data/ci-runner/pypto-toolchain/toolchain/2026.08.5/gcc
export PATH="$GCC15_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$GCC15_ROOT/lib64:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
source "$PTO_EAGER_ROOT/.venv-dsv4-0251rc1/bin/activate"
source "$PTO_EAGER_ROOT/.cache/dsv4-toolchain/atb-package-9.0.0/atb/set_env.sh" --cxx_abi=1
export ASCEND_CUSTOM_OPP_PATH="$PTO_EAGER_ROOT/vllm-ascend-dsv4-pto-0251rc1/.cache/csa/csa-native-ops-install/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$PTO_EAGER_ROOT/vllm-ascend-dsv4-pto-0251rc1/.cache/csa/native-install:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ "$1" == cann92 ]]; then
    export PATH="$PTO_EAGER_ROOT/.cache/dsv4-toolchain/cann92-profiler/tools/profiler/bin:$PATH"
    export LD_LIBRARY_PATH="$PTO_EAGER_ROOT/.cache/dsv4-toolchain/cann92-profiler/tools/profiler/lib64:$LD_LIBRARY_PATH"
fi
