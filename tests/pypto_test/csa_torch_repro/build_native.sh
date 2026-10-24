#!/usr/bin/env bash
# 第二阶段使用：从当前源码独立构建CANN9.2的vLLM-Ascend扩展和自定义算子包。
set -eo pipefail
: "${CSA_WORKSPACE:?先source本机repro-local.sh和模型Python环境}"
: "${CSA_REPO:?源码目录}"
: "${CSA_CANN_ROOT:?CANN9.2安装目录}"
: "${CSA_NATIVE_CMAKE_BIN:?CMake3.31.10的bin目录}"
command -v patch >/dev/null
source "$CSA_CANN_ROOT/set_env.sh"
export CC=${CSA_NATIVE_CC:-/usr/bin/gcc}
export CXX=${CSA_NATIVE_CXX:-/usr/bin/g++}
export PATH="$CSA_NATIVE_CMAKE_BIN:$PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
export CPATH="$CSA_REPO/csrc/third_party/catlass/include${CPATH:+:$CPATH}"
test -d "$CSA_REPO/csrc/third_party/catlass/include"
mkdir -p "$CSA_WORKSPACE/build-logs"

cmake -S "$CSA_REPO" -B "$CSA_WORKSPACE/native-extension-cann92" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${CSA_SOC:-ascend910_9392}" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DASCEND_HOME_PATH="$CSA_CANN_ROOT" \
    -DPYTHON_EXECUTABLE="$(command -v python)" \
    -DPYTHON_INCLUDE_PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    -DTORCH_NPU_PATH="$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch_npu").origin).parent)')" \
    -DCMAKE_INSTALL_PREFIX="$CSA_REPO/.cache/csa/native-install" \
    > "$CSA_WORKSPACE/build-logs/native-extension-configure.log" 2>&1
cmake --build "$CSA_WORKSPACE/native-extension-cann92" --parallel "${CSA_NATIVE_JOBS:-8}" \
    > "$CSA_WORKSPACE/build-logs/native-extension-build.log" 2>&1
cmake --install "$CSA_WORKSPACE/native-extension-cann92" \
    > "$CSA_WORKSPACE/build-logs/native-extension-install.log" 2>&1

ops='compressor;compressor_metadata;vllm_quant_lightning_indexer;vllm_quant_lightning_indexer_metadata;sparse_attn_sharedkv;sparse_attn_sharedkv_metadata;rms_norm_dynamic_quant;inplace_partial_rotary_mul;scatter_nd_update_v2;hc_pre;hc_post;moe_gating_top_k_hash;moe_gating_top_k;dequant_swiglu_quant;add_rms_norm_bias'
(
    # CMake下载GitCode第三方源码时直连，不修改全局代理。
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    cmake -S "$CSA_REPO/csrc" -B "$CSA_WORKSPACE/native-ops-cann92" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DASCEND_COMPUTE_UNIT=ascend910_93 -DASCEND_OP_NAME="$ops" \
        -DVENDOR_NAME=csa_repro -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON \
        -DENABLE_BUILD_PKG=ON -DENABLE_CCACHE=OFF -DENABLE_ASAN=OFF \
        -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$CSA_CANN_ROOT" \
        -DCANN_3RD_LIB_PATH="$CSA_WORKSPACE/deps" \
        > "$CSA_WORKSPACE/build-logs/native-ops-configure.log" 2>&1
    cmake --build "$CSA_WORKSPACE/native-ops-cann92" --target package --parallel "${CSA_NATIVE_JOBS:-8}" \
        > "$CSA_WORKSPACE/build-logs/native-ops-build.log" 2>&1
    cmake --install "$CSA_WORKSPACE/native-ops-cann92" --prefix "$CSA_WORKSPACE/native-ops-install" \
        > "$CSA_WORKSPACE/build-logs/native-ops-install.log" 2>&1
)
python - <<'PY'
import os
from pathlib import Path
repo = Path(os.environ["CSA_REPO"])
for library in (repo / ".cache/csa/native-install").glob("*.so"):
    target = repo / "vllm_ascend" / library.name
    if target.is_symlink():
        target.unlink()
    if target.exists():
        raise FileExistsError(f"已有非链接产物，先确认来源后换新目录：{target}")
    target.symlink_to(library)
PY
test -f "$CSA_WORKSPACE/native-ops-install/packages/vendors/csa_repro_transformer/op_api/lib/libcust_opapi.so"
printf 'CANN9.2 Native依赖构建完成，日志：%s/build-logs\n' "$CSA_WORKSPACE"
