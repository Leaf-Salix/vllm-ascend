#!/usr/bin/env bash
# 在公共 env-dsv4-0251rc1.sh 之后 source；三个 CSA session 共用本入口。
_csa_shared_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readarray -t _csa_shared_values < <(python "$_csa_shared_here/selection.py")
if [[ ${#_csa_shared_values[@]} != 5 ]]; then
    printf '无法读取公共 CSA 工具链配置\n' >&2
    return 1
fi
export CSA_SHARED_TOOLCHAIN_RELEASE=${_csa_shared_values[0]}
export CSA_SHARED_PYPTO_ROOT=${_csa_shared_values[1]}
export CSA_SHARED_PTOAS_ROOT=${_csa_shared_values[2]}
export CSA_SHARED_SIMPLER_ROOT=${_csa_shared_values[3]}
export CSA_SHARED_TOOLCHAIN_MANIFEST=${_csa_shared_values[4]}
if [[ ! -x "$CSA_SHARED_PTOAS_ROOT/bin/ptoas" ]]; then
    printf '公共 PTOAS 尚未构建完成：%s/bin/ptoas\n' "$CSA_SHARED_PTOAS_ROOT" >&2
    return 1
fi
export CSA_STATIC_PYPTO_ROOT="$CSA_SHARED_PYPTO_ROOT" CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export CSA_SDMA_PYPTO_ROOT="$CSA_SHARED_PYPTO_ROOT" CSA_SDMA_SIMPLER_ROOT="$CSA_SHARED_SIMPLER_ROOT"
export HCA_SIMPLER_ROOT="$CSA_SHARED_SIMPLER_ROOT" HCA_SIMPLER_ROOT_ACTIVE="$CSA_SHARED_SIMPLER_ROOT"
export PTOAS_ROOT="$CSA_SHARED_PTOAS_ROOT"
export PATH="$PTOAS_ROOT/bin:$PATH"
export PYTHONPATH="$_csa_shared_here/python_startup${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
unset _csa_shared_here _csa_shared_values
