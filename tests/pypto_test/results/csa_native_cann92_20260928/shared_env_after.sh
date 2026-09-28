#!/usr/bin/env bash
# Shared CSA/HCA release environment. Native custom operators stay on the release.
export PTO_EAGER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# activate first restores _OLD_VIRTUAL_PATH when a venv is already active;
# restore it before cleanup so it cannot reintroduce the old CANN compiler.
if declare -F deactivate >/dev/null; then
    deactivate nondestructive
fi
# The new CANN set_env only removes installations under its own parent. Remove
# previous CANN paths as well when re-sourcing from an existing 9.0 shell.
_dsv4_without_previous_cann() {
    local _dsv4_part _dsv4_joined=""
    local -a _dsv4_parts
    IFS=: read -r -a _dsv4_parts <<< "$1"
    for _dsv4_part in "${_dsv4_parts[@]}"; do
        case "$_dsv4_part" in
            */Ascend/cann/*|*/Ascend/cann-*/*|*/Ascend/ascend-toolkit/*) continue ;;
        esac
        [[ -z "$_dsv4_part" ]] && continue
        _dsv4_joined+="${_dsv4_joined:+:}$_dsv4_part"
    done
    printf '%s' "$_dsv4_joined"
}
export PATH="$(_dsv4_without_previous_cann "${PATH:-}")"
export LD_LIBRARY_PATH="$(_dsv4_without_previous_cann "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(_dsv4_without_previous_cann "${PYTHONPATH:-}")"
export CMAKE_PREFIX_PATH="$(_dsv4_without_previous_cann "${CMAKE_PREFIX_PATH:-}")"
unset -f _dsv4_without_previous_cann
source /data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2/set_env.sh
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
# Same-version local profiler copy satisfies torch_npu's file-owner checks.
export PATH="$PTO_EAGER_ROOT/.cache/dsv4-toolchain/cann92-profiler/tools/profiler/bin:$PATH"
export LD_LIBRARY_PATH="$PTO_EAGER_ROOT/.cache/dsv4-toolchain/cann92-profiler/tools/profiler/lib64:$LD_LIBRARY_PATH"
