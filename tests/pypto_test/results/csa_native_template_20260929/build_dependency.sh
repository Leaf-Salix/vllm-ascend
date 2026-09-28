#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_native_template_20260929"
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
build="$repo/.cache/csa/native-template-cann92-build"
stage="$repo/.cache/csa/native-template-cann92-stage"
# Only the missing release operator; CPU build, no NPU allocation.
cmake -S csrc -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_COMPUTE_UNIT=ascend910_93 -DASCEND_OP_NAME=add_rms_norm_bias \
    -DVENDOR_NAME=csa_template -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON \
    -DENABLE_BUILD_PKG=ON -DENABLE_CCACHE=OFF -DENABLE_ASAN=OFF \
    -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DCANN_3RD_LIB_PATH="$repo/csrc/third_party" \
    -DASCEND_PROTOBUF_SHARED_INCLUDE="$repo/csrc/third_party/ascend_protobuf/src" \
    -DASCEND_PROTOC="$repo/csrc/build/ascend_protobuf_build_transformer-prefix/src/ascend_protobuf_build_transformer-build" \
    > "$root/configure.log" 2>&1
cmake --build "$build" --target package --parallel 8 > "$root/build.log" 2>&1
# The project overrides the configure-time prefix; override again at install.
cmake --install "$build" --prefix "$stage" > "$root/install.log" 2>&1
nm -D "$stage/packages/vendors/csa_template_transformer/op_api/lib/libcust_opapi.so" \
    | rg ' T aclnnAddRmsNormBias(GetWorkspaceSize)?$' > "$root/symbols.txt"
