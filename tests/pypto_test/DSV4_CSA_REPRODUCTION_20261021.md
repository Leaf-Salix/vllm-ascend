# DeepSeek V4 Flash CSA 两阶段复现手册（CANN 9.2）

本手册分两个阶段，均固定 **CANN 9.2.0-beta.2、A3、Python 3.10、Torch 2.10.0 / torch_npu 2.10.0.post2**。

1. **PyTorch 单卡复现**：不安装 vLLM/vLLM-Ascend，不编译 Native 扩展；直接比较当前
   TND-SPMD 与 TND-MPMD 的输出、Top-K、cache/state、图重放及保护区。可选采完整 CSA L1。
2. **安装 vLLM，复现 16 卡 decode**：安装框架、以 CANN 9.2 构建 vLLM-Ascend 自定义算子，
   在本机生成真实权重 KV，再验证 Native、TND-MPMD、TND-SPMD 的实际模型路由和回答。

第一阶段支持随机 checkpoint；第二阶段必须使用完整正式权重。
单卡 NPUGraph 通过不能替代模型图通过；模型检查须证明全部 21 层 CSA 在真实长历史
FULL decode 中重放，不能只看 FULL 标签。两个阶段都关闭 superkernel 与 static kernel。

## 1. 固定版本、发布分支与机器条件

分支用于获取代码，**SHA 才是本次复现版本**。不要只拉移动中的分支头，也不需要各仓分支同名。
下表冻结执行代码；手册和最终证据在其后单独提交，请以发布分支上的本手册为操作说明。

| 仓库/组件 | 公开分支或来源 | 固定提交/版本 |
| --- | --- | --- |
| [vLLM-Ascend](https://github.com/nalinaly/vllm-ascend/tree/dsv4-flash-pto-v0.25.1rc1) | `dsv4-flash-pto-v0.25.1rc1` | `015493995d1f8edc86230c646c30ae5dcb5cf9fb` |
| [PyPTO](https://github.com/nalinaly/pypto/tree/feat/csa-mpmd-upstream-fixes-20261021) | `feat/csa-mpmd-upstream-fixes-20261021` | `d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063` |
| [Simpler](https://github.com/nalinaly/simpler/tree/dsv4-csa-spmd-repro-20261019) | `dsv4-csa-spmd-repro-20261019` | `57e499238fd4ee3548947ad860be456aaeeb16b5` |
| [PTOAS](https://github.com/nalinaly/PTOAS/tree/feat/csa-mpmd-upstream-fixes-20261021) | `feat/csa-mpmd-upstream-fixes-20261021` | `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，CLI 0.69 |
| [PTO-ISA](https://github.com/hw-native-sys/pto-isa) | Simpler `pto_isa.pin` | `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` |
| [vLLM](https://github.com/vllm-project/vllm)（第二阶段） | 官方仓指定提交 | `752a3a504485790a2e8491cacbb35c137339ad34` |
| LLVM/MLIR | `vpto-dev/llvm-project` | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42` + 下文单行头文件补丁 |
| CANN CMake 工具 | `gitcode.com/cann/cmake` | `a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70` |
| CATLASS（第二阶段） | vLLM-Ascend 的 Git 子模块 | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| CANN 构建与运行 | 预先安装 SDK/驱动 | **9.2.0-beta.2** |
| ATB（第二阶段） | 与 CANN 配套安装，C++ ABI=1 | **9.2.0-beta.2** |
| PyPTO/PTOAS Host 编译器 | GCC / CMake | 15.2.0 / 4.4.2 |
| Native Host 编译器（第二阶段） | GCC / CMake / GNU patch | 10.3.1 / 3.31.10 / 2.7.6 |
| Triton Ascend（第二阶段） | 官方 Ascend Python 包源 | 3.2.2 |

硬件为 Linux aarch64 / Ascend A3，单设备 24 AIC + 48 AIV；第二阶段需同机 16 张可互联的卡。
已测驱动为 26.0.rc1。准备 Python 3.10 开发头文件、git、make、curl、xz、GNU patch。
GCC 15 用于 PTO 工具链；Native 构建使用上表独立的 Host 编译器，避免第三方 abseil 的头文件兼容问题。

正式权重为 [DeepSeek-V4-Flash-0731-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8)。
第一阶段读取第 4 层对应分片；第二阶段需完整 75 个 ModelSlim 分片、配置及 tokenizer。
不需要原作者的 KV 缓存或 bundle；第二阶段说明如何自行生成 KV。

## 2. 新目录中浅拉取源码

在同一 Bash 会话按顺序执行。使用新的工作目录，以下目录名与已提交的公共 SDK 清单一致。
`.worktrees` 中放普通 clone 即可，各构建产物使用下文指定的独立目录。

```bash
set -eo pipefail
export CSA_WORKSPACE=/your/workspace/csa-repro
mkdir -p "$CSA_WORKSPACE/.worktrees" "$CSA_WORKSPACE/deps" "$CSA_WORKSPACE/results"
cd "$CSA_WORKSPACE"
fetch_pinned() {
    local url=$1 revision=$2 destination=$3
    test ! -e "$destination" || { printf '目录已存在：%s\n' "$destination" >&2; return 1; }
    git init -q "$destination"
    git -C "$destination" remote add origin "$url"
    git -C "$destination" fetch --depth=1 origin "$revision"
    git -C "$destination" checkout --detach FETCH_HEAD
}
fetch_pinned https://github.com/nalinaly/vllm-ascend.git 015493995d1f8edc86230c646c30ae5dcb5cf9fb \
  "$CSA_WORKSPACE/vllm-ascend-dsv4-pto-0251rc1"
fetch_pinned https://github.com/nalinaly/pypto.git d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063 \
  "$CSA_WORKSPACE/.worktrees/pypto-csa-shared-group-dims-20261024"
fetch_pinned https://github.com/nalinaly/simpler.git 57e499238fd4ee3548947ad860be456aaeeb16b5 \
  "$CSA_WORKSPACE/.worktrees/simpler-csa-core-args-256-20261016"
fetch_pinned https://github.com/nalinaly/PTOAS.git 914863ed3d0eb3e92a4433a4a706a86cabf4c91f \
  "$CSA_WORKSPACE/.worktrees/ptoas-csa-mpmd-upstream-20261021"
export CSA_REPO="$CSA_WORKSPACE/vllm-ascend-dsv4-pto-0251rc1"
export CSA_PYPTO="$CSA_WORKSPACE/.worktrees/pypto-csa-shared-group-dims-20261024"
export CSA_SIMPLER="$CSA_WORKSPACE/.worktrees/simpler-csa-core-args-256-20261016"
export CSA_PTOAS="$CSA_WORKSPACE/.worktrees/ptoas-csa-mpmd-upstream-20261021"
git -C "$CSA_PYPTO" config submodule.simpler.url https://github.com/nalinaly/simpler.git
git -C "$CSA_PYPTO" submodule update --init --recursive --depth=1
mkdir -p "$CSA_SIMPLER/build"
fetch_pinned https://github.com/hw-native-sys/pto-isa.git 327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d \
  "$CSA_SIMPLER/build/pto-isa"
(
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    fetch_pinned https://gitcode.com/cann/cmake.git a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70 \
      "$CSA_WORKSPACE/deps/cann-cmake"
)
```

此时只获取 vLLM-Ascend 仓库中的 PTO 源码，不安装该 Python 项目。
PyPTO 的 Simpler 子模块和实际使用的 Simpler 都须对应上表提交。
不在任务排队或执行时修改已冻结源码、切换 SDK 或原位重建 SDK。

## 第一阶段：PyTorch 单卡

### 3. 本机路径和独立 Python 环境

```bash
cat > "$CSA_WORKSPACE/repro-local.sh" <<'BASH'
export CSA_CANN_ROOT=/path/to/cann-9.2.0-beta.2
# 第二阶段还需要对应版本的ATB；目录下应有atb/set_env.sh。
export CSA_ATB_ROOT=/path/to/nnal/atb/9.2.0-beta.2
export CSA_GCC15=/path/to/gcc-15.2.0
export CSA_MODEL=/path/to/DeepSeek-V4-Flash-0731-w8a8
export CSA_LLVM_JOBS=8 CSA_PTOAS_JOBS=8 PYPTO_BUILD_JOBS=8
# 第二阶段填写本机的16卡通信网卡和IPv4地址。
export CSA_HOST=YOUR_HOST_IPV4
export CSA_NIC=YOUR_NETWORK_INTERFACE
export CSA_SOC=ascend910_9392
export CSA_NATIVE_CC=/usr/bin/gcc CSA_NATIVE_CXX=/usr/bin/g++
export CSA_NATIVE_JOBS=8
BASH
source "$CSA_WORKSPACE/repro-local.sh"
unset PYTHONPATH LD_LIBRARY_PATH ASCEND_CUSTOM_OPP_PATH
source "$CSA_CANN_ROOT/set_env.sh"
python3.10 -m venv "$CSA_WORKSPACE/.venv"
source "$CSA_WORKSPACE/.venv/bin/activate"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export NO_PROXY="pypi.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org,mirrors.huaweicloud.com${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
python -m pip install \
  torch==2.10.0 torch-npu==2.10.0.post2 numpy==2.2.6 safetensors==0.8.0 \
  cloudpickle==3.1.1 pybind11==3.1.0 pyyaml==6.0.3 psutil==6.0.0 tabulate==0.9.0 \
  attrs==24.2.0 decorator==5.1.1 scipy==1.13.1 protobuf==7.36.2 \
  cmake==4.4.2 ninja==1.13.0 nanobind==2.15.0 scikit-build-core==1.0.3 \
  setuptools==79.0.1 setuptools-scm==10.2.3 wheel==0.48.0
```

没有正式权重时，可在 CPU 上生成固定随机 checkpoint：

```bash
python "$CSA_REPO/tests/pypto_test/csa_torch_repro/generate_random_checkpoint.py" \
  --output "$CSA_WORKSPACE/random-csa-weights"
```

只在第一阶段将 `CSA_MODEL` 改为该目录。它包含固定 seed 的最小单层数据，不读取模型文件，
不能用于第二阶段整模型推理。随机数据的 Top-K/压缩/访存不同，耗时不能当作正式权重性能。

## 4. 构建 PTO 工具链

### 4.1 LLVM / PTOAS

```bash
source "$CSA_CANN_ROOT/set_env.sh"
export CC="$CSA_GCC15/bin/gcc-15" CXX="$CSA_GCC15/bin/g++-15"
export PATH="$CSA_GCC15/bin:$PATH"
export LD_LIBRARY_PATH="$CSA_GCC15/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

fetch_pinned https://github.com/vpto-dev/llvm-project.git \
  fa2fd1f75d742fec91ecf40d98c9c7961647ec42 "$CSA_WORKSPACE/deps/llvm-project"
git -C "$CSA_WORKSPACE/deps/llvm-project" apply --unidiff-zero <<'PATCH'
diff --git a/mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h b/mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h
--- a/mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h
+++ b/mlir/include/mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h
@@ -12 +12,2 @@
 #include "mlir/Support/LLVM.h"
+#include <cstdint>
PATCH
cmake -G Ninja -S "$CSA_WORKSPACE/deps/llvm-project/llvm" -B "$CSA_WORKSPACE/deps/llvm-build" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=AArch64 -DBUILD_SHARED_LIBS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF -DMLIR_INCLUDE_EXAMPLES=OFF \
  -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF
cmake --build "$CSA_WORKSPACE/deps/llvm-build" --parallel "${CSA_LLVM_JOBS:-2}" --target \
  mlir-tblgen MLIRMlirOptMain MLIRBufferizationTransforms MLIRArithTransforms \
  MLIRTensorTransforms MLIRFuncTransforms MLIRIR MLIRParser MLIRPass \
  MLIREmitCDialect MLIRSCFDialect MLIRGPUDialect MLIRTransforms MLIRTargetCpp \
  MLIRAffineDialect MLIRArithDialect MLIRTensorDialect MLIRControlFlowDialect \
  MLIRBufferizationDialect MLIRFuncDialect MLIRFuncInlinerExtension \
  MLIRLLVMIRTransforms MLIRSupport MLIRTransformUtils MLIRAnalysis \
  MLIRLLVMDialect MLIRMemRefDialect MLIRMathDialect LLVMOption LLVMCore LLVMObject LLVMSupport
cmake -G Ninja -S "$CSA_PTOAS" -B "$CSA_PTOAS/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$CSA_WORKSPACE/deps/llvm-build/lib/cmake/llvm" \
  -DMLIR_DIR="$CSA_WORKSPACE/deps/llvm-build/lib/cmake/mlir" \
  -DCANN_3RD_LIB_PATH="$CSA_WORKSPACE/deps" -DPTO_ENABLE_PYTHON_BINDING=OFF
cmake --build "$CSA_PTOAS/build" --target ptoas-native --parallel "${CSA_PTOAS_JOBS:-2}"
export PTOAS_ROOT="$CSA_PTOAS/build"
export PATH="$PTOAS_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CSA_WORKSPACE/deps/llvm-build/lib:$LD_LIBRARY_PATH"
"$PTOAS_ROOT/bin/ptoas" --version
```

若已有同提交、同构建选项的 LLVM，可复用其构建目录并相应设置 `LLVM_DIR/MLIR_DIR`；
不要拿其他源码树的 PTOAS `.so` 或可执行文件直接冒充本次构建。
该 LLVM 头文件使用 `int64_t`，在 GCC 15 下需要显式包含 `<cstdint>`。
LLVM 的完整复现版本为固定 SHA 加上面的单行补丁。
这里列出该 PTOAS native CLI 的 LLVM/MLIR 组件，构建系统会补齐其传递依赖。
关闭测试、示例和 ZSTD 与原环境一致；不要求构建未使用的 SPIR-V 反序列化器或所有 LLVM
工具。完整 `ptoas-native` 的链接及后续 CSA 编译是这些组件是否齐备的实际检查。

### 4.2 Simpler 与 PTO-ISA

```bash
python -m venv "$CSA_SIMPLER/.venv"
"$CSA_SIMPLER/.venv/bin/python" -m pip install \
  nanobind==2.15.0 scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0 cloudpickle==3.1.1
(
    cd "$CSA_SIMPLER"
    CMAKE_BUILD_PARALLEL_LEVEL=2 "$CSA_SIMPLER/.venv/bin/python" -m pip install \
      --no-build-isolation --no-deps -e .
)
test "$(cat "$CSA_SIMPLER/pto_isa.pin")" = \
  327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d
test "$(git -C "$CSA_SIMPLER/build/pto-isa" rev-parse HEAD)" = \
  327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d
```

PTO-ISA 在该流程中作为固定版本头文件使用；Simpler 会校验并记录它，不需要另外安装一个
名称相近的 PyPI 包。Simpler 的 256 Tensor + 16 scalar ABI 必须与 PyPTO 的 runtime 子模块
一致，不能混用系统里另一个 `_task_interface.so`。

### 4.3 PyPTO

```bash
(
    cd "$CSA_PYPTO"
    source .claude/skills/testing/load-env.sh
    cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DPYPTO_DEBUG_INFO_LEVEL=1 -DPYPTO_BUILD_TORCH_NPU=ON \
      -DPython_EXECUTABLE="$(command -v python)" \
      -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"
    cmake --build build --parallel "$PYPTO_BUILD_JOBS"
)
```


### 5. 单卡入口与验收

保存运行环境。`bootstrap.py --shared-toolchain` 按源码仓公共 manifest 绑定 SDK；
用命名空间加载 PTO 源码，禁止导入推理引擎和 Native 扩展。

```bash
cat > "$CSA_WORKSPACE/env-csa-torch.sh" <<'BASH'
export CSA_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CSA_WORKSPACE/repro-local.sh"
unset PYTHONPATH LD_LIBRARY_PATH ASCEND_CUSTOM_OPP_PATH
source "$CSA_CANN_ROOT/set_env.sh"
source "$CSA_WORKSPACE/.venv/bin/activate"
export CSA_REPO="$CSA_WORKSPACE/vllm-ascend-dsv4-pto-0251rc1"
export CSA_TESTS="$CSA_REPO/tests/pypto_test"
export PATH="$CSA_GCC15/bin:$PATH"
export LD_LIBRARY_PATH="$CSA_GCC15/lib64:$CSA_WORKSPACE/deps/llvm-build/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
BASH
cat > "$CSA_WORKSPACE/run-csa-torch.sh" <<'BASH'
#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env-csa-torch.sh"
case_name=${1:?C04或C08等场景编号}
label=${2:?使用新的输出标签}
shift 2
: "${TASK_DEVICE:?由设备调度服务分配单张卡}"
out="$CSA_WORKSPACE/results/${case_name}_${label}"
export PYPTO_PROG_BUILD_DIR="$out-build" PYPTO_CACHE_DIR="$out-cache"
export ASCEND_WORK_PATH="$out-ascend" ASCEND_PROCESS_LOG_PATH="$out-ascend/log" ASCEND_CACHE_PATH="$out-ascend/cache"
mkdir -p "$PYPTO_PROG_BUILD_DIR" "$PYPTO_CACHE_DIR" "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
exec python "$CSA_TESTS/csa_torch_repro/bootstrap.py" --shared-toolchain --workspace-root "$CSA_WORKSPACE" \
  "$CSA_TESTS/csa_torch_repro/current.py" --case "$case_name" --model "$CSA_MODEL" --output "$out" "$@"
BASH
printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-csa-torch.sh" C04 first --profile-windows 10
task-submit --device auto --max-time 3600 "$CSA_COMMAND"
# C04完成后，可补混长；不需要L1时省略--profile-windows。
printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-csa-torch.sh" C08 first
task-submit --device auto --max-time 3600 "$CSA_COMMAND"
```

`task-submit` 是本机队列，不是 PyPTO 的依赖。其他机器使用自己的调度器分配卡，
设置 `TASK_DEVICE` 后运行脚本。通过 `--status <task-id>`、`--log <task-id>` 等待终态；
输出目录已存在时入口拒绝覆盖。

- **C04**：128K/B16/S6，16 个请求各 6 token，T=96。
- **C08**：128K/B16 混长，query 固定为 `[1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]`，T=56、容量也为 56。
- 其他 C01～C08 的业务描述见 [cases.json](csa_torch_repro/cases.json) 及
  [性能规范](../../vllm_ascend/ops/pypto/CSA_PERFORMANCE_TEST_SPEC.md)。历史 H 不包含本轮 query。

输入准备在 PyTorch 中完成：固定 seed=1024、正式第 4 层/NZ2、独立页表和随机历史
cache/state。Indexer 每页为 4096 字节 key + 64 字节 FP16 scale，状态保留物理步长和保护区。
这不使用或生成第二阶段的真实模型 KV bank。

检查 `result.json`：

- 退出码 0，`stage=complete`、`passed=true`；CPU 解析通过不算设备验收。
- 两种实现的诊断入口核对 output/Top-K/score/cache/state；普通入口验证无外部 scratch，
  只有 55 个外部 Tensor，不写测试专用 Top-K/score 缓冲。
- eager、两次 NPUGraph replay、保护区全部通过；浮点按 JSON 中明确容差检查，整数/索引与保护区精确一致，另记 bit 对照。
- `dependency_check` 为 `vllm_imported=false`、`native_custom_ops_registered=false`。
- 可选 L1 采集时每侧至少 10 个窗口，报告 min/median/mean/max 和全部样本，保存
  `l1_00_spmd/swimlane.json` 等文件，并在采样后再核对状态。SPMD 的内部信号清零也计入完整 L1。

完整 CSA 是 HC_pre → input RMSNorm → CSA → HC_post，L1 取首个 incore 开始至最后一个结束，
保留 task 间空隙；不累加重叠 task。编译、warmup、测试输入恢复和 D2H 不纳入计时。
本入口两种实现使用同一套 PyTorch 输入，不把其结果冒充 Native 精度或完整六列性能表。

## 第二阶段：安装 vLLM 并复现 16 卡 decode

### 6. 框架与 Python 环境

第二阶段恢复正式权重路径；随机 checkpoint 不能用于这一阶段。
使用单独 `.venv-model`，第一阶段环境保留不变。

```bash
source "$CSA_WORKSPACE/repro-local.sh"
fetch_pinned https://github.com/vllm-project/vllm.git \
  752a3a504485790a2e8491cacbb35c137339ad34 "$CSA_WORKSPACE/vllm"
(
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    git -C "$CSA_REPO" -c http.proxy= submodule update --init --depth=1 csrc/third_party/catlass
)
python3.10 -m venv "$CSA_WORKSPACE/.venv-model"
source "$CSA_WORKSPACE/.venv-model/bin/activate"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pip install --no-deps -r "$CSA_REPO/tests/pypto_test/csa_torch_repro/model-requirements.txt"
python -m pip install --no-deps triton-ascend==3.2.2 \
  --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1 \
  python -m pip install --no-build-isolation --no-deps -e "$CSA_WORKSPACE/vllm"
COMPILE_CUSTOM_KERNELS=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1rc1 \
  python -m pip install --no-build-isolation --no-deps -e "$CSA_REPO"
python -m pip install --target "$CSA_WORKSPACE/native-build-tools" cmake==3.31.10
export CSA_NATIVE_CMAKE_BIN="$CSA_WORKSPACE/native-build-tools/cmake/data/bin"
```

`model-requirements.txt` 固定已验证的离线 decode Python 依赖集合；两个源码项目用
`--no-deps` 安装，避免项目的历史依赖约束自动替换 Torch 等版本。
Triton Ascend 使用[官方安装指南指定的包源](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md)，
不能把默认 PyPI 镜像找不到 3.2.2 误判为无需安装。
`VLLM_TARGET_DEVICE=empty` 避免构建 CUDA 扩展；实际 NPU 后端由 vLLM-Ascend 提供。

### 7. 以 CANN 9.2 构建 Native 依赖

这里编译的是 **vLLM-Ascend 仓库的 C++ 扩展和自定义 AscendC 算子**，不是重建 CANN。
这些依赖用于 HCA、模型其他算子及 Native 对照；即使 CSA 使用 PTO 也仍需它们。

```bash
source "$CSA_CANN_ROOT/set_env.sh"
source "$CSA_WORKSPACE/.venv-model/bin/activate"
command -v patch
export CSA_NATIVE_CMAKE_BIN="$CSA_WORKSPACE/native-build-tools/cmake/data/bin"
bash "$CSA_REPO/tests/pypto_test/csa_torch_repro/build_native.sh"
```

脚本明确给 CMake 指定同一 `CSA_CANN_ROOT`，在独立目录构建，使用系统 GCC 10.3.1。
它安装 C++ 扩展到源码仓 `.cache/csa/native-install`，链接到 Python 包目录；
自定义算子包在 `$CSA_WORKSPACE/native-ops-install/packages/vendors/csa_repro_transformer`。
包含模型所需 Compressor、Indexer、稀疏注意力、HC、MoE 和 AddRmsNormBias。
构建日志在 `$CSA_WORKSPACE/build-logs/`；构建失败时不能继续把旧库当作新产物使用。

### 8. 模型运行环境

第二阶段还必须安装 **ATB 9.2.0-beta.2**。当前 Torch 使用 C++ ABI=1，显式加载对应目录；
缺少它时 torch_npu 的 ATB 扩展会在 worker 初始化阶段报告 `libatb.so` 找不到，尚未执行 CSA。

```bash
cat > "$CSA_WORKSPACE/env-csa-model.sh" <<'BASH'
export CSA_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CSA_WORKSPACE/repro-local.sh"
unset PYTHONPATH LD_LIBRARY_PATH ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH ATB_OPSRUNNER_PATH ATB_LIB_PATH
source "$CSA_CANN_ROOT/set_env.sh"
source "$CSA_ATB_ROOT/atb/set_env.sh" --cxx_abi=1
source "$CSA_WORKSPACE/.venv-model/bin/activate"
export CSA_REPO="$CSA_WORKSPACE/vllm-ascend-dsv4-pto-0251rc1"
export CSA_TESTS="$CSA_REPO/tests/pypto_test"
export PATH="$CSA_GCC15/bin:$PATH"
export ASCEND_CUSTOM_OPP_PATH="$CSA_WORKSPACE/native-ops-install/packages/vendors/csa_repro_transformer"
export LD_LIBRARY_PATH="$CSA_REPO/.cache/csa/native-install:$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$CSA_GCC15/lib64:$CSA_WORKSPACE/deps/llvm-build/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$CSA_WORKSPACE/vllm:$CSA_REPO:$CSA_TESTS${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true OMP_PROC_BIND=false OMP_NUM_THREADS=10
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export TORCH_EXTENSIONS_DIR="$CSA_WORKSPACE/torch_extensions"
source "$CSA_TESTS/csa_shared_toolchain/env.sh"
BASH
# 公共SDK选择器用这个入口定位工作区；不从其他旧检出选择依赖。
cat > "$CSA_WORKSPACE/env-dsv4-0251rc1.sh" <<'BASH'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env-csa-model.sh"
BASH
source "$CSA_WORKSPACE/env-csa-model.sh"
```

`CSA_HOST` 与 `CSA_NIC` 必须填写本机可用 IPv4/网卡；不要复制原作者机器的地址。
后续所有 Native、MPMD、SPMD 比较使用同一环境、权重、KV bank 和确定性设置。

### 9. 在本机生成和审计真实 KV

本地已有、且来源匹配的 bank 可以只读复用；其他复现者应自行执行以下流程，
不能拿第一阶段的随机 cache 代替。先生成 128K 公共前缀，再对 40 个不同问题的后缀做
真实 Native prefill，得到每个问题独立的 target/draft cache。
公共前缀沿用 Native TP4/DP4、ND 权重；后缀以 TP1/DP16、NZ2 生成，正式 decode 三侧也统一 NZ2。

```bash
export CSA_KV_ROOT="$CSA_WORKSPACE/kv"
mkdir -p "$CSA_KV_ROOT"
python "$CSA_TESTS/offline_pd/run.py" plan --model "$CSA_MODEL" \
  --bank "$CSA_KV_ROOT/prefix128k" --histories 131072

cat > "$CSA_WORKSPACE/run-kv-job.sh" <<'BASH'
#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env-csa-model.sh"
job=${1:?prefix或suffix}
shift
# task-submit可能附加设备参数；设备映射由TASK_DEVICE统一提供。
if [[ ${1:-} == --device ]]; then shift 2; fi
(( $# == 0 )) || exit 2
root="$CSA_WORKSPACE/kv"
out="$root/${job}_run"
export ASCEND_WORK_PATH="$out-ascend" ASCEND_PROCESS_LOG_PATH="$out-ascend/log" ASCEND_CACHE_PATH="$out-ascend/cache"
export VLLM_CACHE_ROOT="$out-vllm-cache" TRITON_CACHE_DIR="$out-triton-cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
case "$job" in
prefix)
    python "$CSA_TESTS/offline_pd/run.py" prefill --model "$CSA_MODEL" \
      --bank "$root/prefix128k" --output "$out" --host "$CSA_HOST" --nic "$CSA_NIC" \
      --backend native --gpu-memory-utilization 0.92 --weight-nz-mode 0 --deterministic --event-work-mode 1
    ;;
suffix)
    python "$CSA_TESTS/offline_pd/run.py" decode --model "$CSA_MODEL" \
      --bank "$root/mixed128k" --save-extended-bank "$root/mixed128k_fixed" --output "$out" \
      --host "$CSA_HOST" --nic "$CSA_NIC" --backend native --decode-dp 16 --batch 1 --mixed-requests \
      --gpu-memory-utilization 0.92 --decode-tokens 1 --graph-mode eager --no-static-kernel \
      --weight-nz-mode 2 --deterministic --event-work-mode 1
    ;;
*) exit 2 ;;
esac
BASH
printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-kv-job.sh" prefix
task-submit --device auto --device-num 16 --max-time 7200 "$CSA_COMMAND"
```

前缀任务完成且 exit=0 后再执行下面的 CPU 审计与问题准备：

```bash
python "$CSA_TESTS/offline_pd/run.py" audit --model "$CSA_MODEL" --bank "$CSA_KV_ROOT/prefix128k"
python "$CSA_TESTS/low_acceptance_20261001/prepare_mixed.py" \
  --source-bank "$CSA_KV_ROOT/prefix128k" --bank "$CSA_KV_ROOT/mixed128k" \
  --encoding "$CSA_WORKSPACE/vllm/vllm/tokenizers/deepseek_v4_encoding.py" \
  --record "$CSA_KV_ROOT/mixed128k_questions.json"
python "$CSA_TESTS/low_acceptance_20261001/fixed_bank.py" prepare \
  --source "$CSA_KV_ROOT/mixed128k" --bank "$CSA_KV_ROOT/mixed128k_fixed"
printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-kv-job.sh" suffix
task-submit --device auto --device-num 16 --max-time 7200 "$CSA_COMMAND"
```

后缀任务完成后，只做一次完整性审计：

```bash
python "$CSA_TESTS/low_acceptance_20261001/fixed_bank.py" audit --bank "$CSA_KV_ROOT/mixed128k_fixed"
```

必须得到 `audit.json: status=PASS`，40 个不同问题的长度、owner、target/draft 覆盖和
cache 结构完整。后续所有实现只读恢复同一 bank；不要每轮重新 prefill 或反复 hash。
`plan.json` 记录本机模型路径、输入及长度，`mixed128k_questions.json` 保存问题原文。
不要手工给一个未审计的 bank 写 PASS，也不要改已确定的问题去筛选更高接受率。

### 10. 冻结模型验证源码并运行 16 卡

当前模型验证保留两档图，并核对满批、尾批和 padding 的真实重放。
先完成第一阶段；仓内 `graph_replay_service_validation.json` 是该发布版本已完成的
服务级单卡证据，不能把它说成接收方已经完成了模型验证。

```bash
export CSA_VALIDATION="$CSA_TESTS/csa_tnd_model_validation_20261023"
python "$CSA_VALIDATION/prepare.py" repro_cann92 --graph-replay-fix
export CSA_FROZEN="$CSA_VALIDATION/experiments/repro_cann92/source"
export CSA_GATE="$CSA_VALIDATION/graph_replay_service_validation.json"

# 三个命令分别提交；每次都保留独立输出目录，不能覆盖旧结果。
for variant in native mpmd_tnd spmd_tnd; do
    printf -v CSA_COMMAND '%q ' bash \
      "$CSA_FROZEN/tests/pypto_test/csa_tnd_model_validation_20261023/run.sh" \
      "$CSA_FROZEN" "$CSA_WORKSPACE/results/model_${variant}_b16_eos2048" "$variant" 16 0.92 "$CSA_GATE" \
      --environment "$CSA_WORKSPACE/env-csa-model.sh" --bank "$CSA_KV_ROOT/mixed128k_fixed" \
      --vllm-root "$CSA_WORKSPACE/vllm" --host "$CSA_HOST" --nic "$CSA_NIC" \
      --ring-heap-mb 16,256,16,16 --ring-task-window 4096 --hccl-buffer-mb 1800 \
      --decode-tokens 2048 --respect-eos
    task-submit --device auto --device-num 16 --max-time 5400 "$CSA_COMMAND"
done
```

固定配置为 TP1、DP=EP16、单卡 B16、DSpark 出5验6、正常EOS停止、每请求最多2048 token、NZ2、EPLB关闭，
21 层 CSA 切换实现，20 层 HCA 始终 Native；确定性 level=1、HCCL确定性、AIV、event mode=1。
模型采用 npugraph_ex/FULL_DECODE_ONLY，但不启用 static kernel 或 superkernel。
临时工作区由 Simpler 内部管理，heap `[16,256,16,16] MiB`、task window 4096。

启动器检查 SDK 与门禁后运行模型，自动执行 `check.py`。验收必须包括：

- 16 个 rank 都完成，各 16 个真实请求，输出长度在1～2048之间、token在有效词表范围内，零抢占。
- `sampling.ignore_eos=false`；逐条保存`finished/finish_reason/stop_reason`及实际输出全文。
  `stop`与`length`分别统计，达到2048上限的请求不能当作完整答案通过。
- PTO路径实际为21层TND CSA + 20层Native HCA，所有层外部PTO工作区为0。
- 真实长历史 FULL 步骤同时覆盖满批、尾批、padding；21层全部放行图重放，且没有Python重提交。
- 首个S1步骤由框架选择NONE/eager，短历史空闲补齐调用单独记录，不冒充真实图重放。
- 不能仅看到 `FULL`、`Replaying aclgraph` 或无报错就标模型验证通过。

B16三侧及输出对照通过后，可按同一命令测试B24：替换 batch为24、预算0.935、
HCCL缓冲768，输出目录改为 `model_${variant}_b24_eos2048`。三侧策略与输入保持一致，
不能用抢占或减少实际请求数绕过显存不足。其他档位不在本次整模型验收范围。

### 11. 结果、完整回答与精度边界

```bash
# 内容不通过时compare会非零退出；仍导出已生成的失败报告供审查，不能改标PASS。
CSA_COMPARE_EXIT=0
python "$CSA_VALIDATION/compare.py" \
  "$CSA_WORKSPACE/results/model_native_b16_eos2048" \
  "$CSA_WORKSPACE/results/model_spmd_tnd_b16_eos2048" \
  "$CSA_WORKSPACE/results/model_mpmd_tnd_b16_eos2048" \
  --output "$CSA_WORKSPACE/results/b16_eos2048_comparison.json" || CSA_COMPARE_EXIT=$?
test -f "$CSA_WORKSPACE/results/b16_eos2048_comparison.json"
printf '答案对照退出码：%s（非零表示未通过，详见JSON）\n' "$CSA_COMPARE_EXIT"
python "$CSA_VALIDATION/export_qa.py" "$CSA_WORKSPACE/download/CSA_TND_B16_EOS2048" \
  --comparisons "$CSA_WORKSPACE/results/b16_eos2048_comparison.json"
```

三侧应在本机使用相同源码、SDK、权重、KV和推理配置，不用历史另一环境的结果填数。
简单重复句子要求正常停止且完整内容/重复次数正确，再比较三侧token；复杂问题、首处分歧和DSpark接受事件完整披露。
数值、账本和状态迁移题按题意计算标准结果，Native输出也接受同一检查。正常EOS、内容正确和格式合规分别报告；
解析不完整的答案标为待人工核对，不能直接算对或算错。不同长度的输出按完整长度统计差异，不能只比较共同前缀。
正常EOS会改变每个请求的生成长度及接受统计，不能与固定192token的接受率混为同一基准；不修改问题去凑目标接受率。
当前PTO与Native并非全bit/全token一致，复杂问题接受率也不是一方始终更优。
MPMD/SPMD彼此以及图修复前后PTO输出回归，要与Native精度差异分开记录。
HTML/Markdown保留全部实际生成的答案、结束原因及题意核对；JSON保留原始token、标准结果和来源。
达到2048上限的答案会明确标为未证明完整，不能用旧192-token前缀验证代替本项。

## 12. 已验证内容、问题与证据

- 当前SDK包含内部工作区所需Group动态维度修复；不再让普通调用在Python申请每层scratch或Top-K。
- 修正了旧模型dummy历史过短导致捕获Native、真实decode回退eager的问题。
- Compact metadata固定容量，由device长度屏蔽失活请求；保留页表、Q/L、KV根地址与策略检查，支持尾批和padding。
- 单卡脚本保留全部图及输入/binding生命周期；过早释放上一档图可能在下一档capture_begin触发图池断言。
- 第一阶段当前入口的验证记录、第二阶段构建与运行来源见下方证据；未测配置不标通过。

本次独立环境的结果如下；表中通过范围按实际证据填写。

| 阶段/对象 | 验证范围 | 结果 |
| --- | --- | --- |
| PyTorch 单卡 C04 | 两种TND、诊断/普通入口、两次NPUGraph、保护区、各10窗完整L1 | PASS；已对照数值逐bit一致、外部scratch为0 |
| PyTorch 单卡 C08 | 固定混长T56、两种TND、诊断/普通入口、两次NPUGraph、保护区 | PASS；已对照数值逐bit一致、外部scratch为0 |
| CANN9.2 Native构建 | C++扩展、15个自定义AscendC算子、打包与安装 | PASS |
| Native 16卡 B16（固定192token） | 256请求、真实S6、零抢占 | PASS；仅作功能证据 |
| TND-MPMD 16卡 B16（固定192token） | 每rank满批33步、尾批/padding15步、21层实际图重放 | PASS；零抢占、外部scratch为0 |
| TND-SPMD 16卡 B16（固定192token） | 每rank满批33步、尾批/padding15步、21层实际图重放 | PASS；零抢占、外部scratch为0 |

单卡证据见[当前两种TND的纯PyTorch验收记录](csa_torch_repro/verification_single_card_20261024.json)。
本轮新建两套Python环境和Native构建产物；按公共工具链要求复用同一套已构建的
PyPTO/PTOAS/Simpler，未在有排队任务时重建或替换共享SDK。模型验证只读复用本地
已审计KV，本轮没有重复prefill；接收方按第9节自行生成。此处不报告B24在新构建下已通过。
正常EOS/2048token三方验证也已完成，16卡执行均exit=0、零抢占。按去重后的16个问题统计：

| 实现 | 正常停止 | 到达2048上限 | 确认完整且内容正确 | 两道重复题实际次数（要求80/81） |
| --- | --- | --- | --- | --- |
| Native | 11 | 5 | 0 | 85 / 83 |
| TND-MPMD | 9 | 7 | 0 | 82 / 83 |
| TND-SPMD | 9 | 7 | 0 | 82 / 83 |

“0”表示本次规则没有确认任何整题完整且内容正确；截断和解析不完整也不计通过，不能将它们一概认定为数值错误。
三侧简单重复题的完整内容校验确实失败，`compare.py`在保存报告后exit=1；不放宽门槛。
旧192token的简单题PASS只证明生成前缀，不能解释为完整答案正确。

两种PTO的全部354144个输出token、256个请求的DSpark接受事件一致；每个rank均覆盖43个满批、
350个尾批步骤（其中349个padding），21层CSA全部实际图重放，无Python重提交，外部scratch为0。
Native共输出297520token；与任一PTO完整token串一致的请求为48/256，位置差异计数含长度差，
不将其直接解释为数值错误数。正常EOS下平均接受长度为Native 4.367100、两种PTO 4.371089。

证据见[全9.2环境固定192token功能记录](csa_torch_repro/verification_model_20261024.json)与
[正常EOS/2048完整答案记录](csa_torch_repro/verification_eos2048_20261024.json)。
本机完整问题、实际回答、结束原因、题意标准结果及原始token位于：
`/data/pyptouser/qinchuanyu/pto-eager/downloads/CSA_TND_CANN92_B16_EOS2048_20261024/`。
接收方执行第11节会在自己的`download/CSA_TND_B16_EOS2048/`生成同类文件。

详细历史与问题状态分别见[主验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)、
[问题状态表](CSA_THREE_SESSIONS_ISSUE_STATUS.md)。
