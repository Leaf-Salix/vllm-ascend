# DeepSeek V4 Flash CSA PyTorch 单卡复现手册

本手册使用 **PyTorch + torch_npu + PyPTO** 直接运行 TND CSA，比较冻结的单 SPMD
与多 task MPMD。无需安装 vLLM、vLLM-Ascend、ATB，也无需编译 Native 自定义算子。
算子源码目前存放在 `nalinaly/vllm-ascend` 仓库；下文只从该仓获取 PTO 源码和测试工具。

单卡入口支持真实权重和本地随机 checkpoint：真实权重读取模型第 4 层；
**没有权重时按第 3.1 节生成测试数据，同样可以上卡、运行图重放并导出泳道。**
Q/L、RoPE、页表、slots 及 cache/state 均由 PyTorch 构造。
两侧使用同一输入和初态，检查 eager、NPUGraph replay、采样后状态，并记录完整 CSA L1。
不启用 superkernel 或 npugraph_ex static kernel；PyPTO/PTOAS/CCEC 的算子编译仍需执行。

## 1. 固定版本与前提

| 项目 | 版本或提交 |
| --- | --- |
| PTO 算子与单卡工具源码 | `nalinaly/vllm-ascend`，`649d6214de8f247883520d6f9df8762b7ee7876e` |
| TND 单 SPMD 参考 | `84751e34cb1f27eff9e4d3fdf03f8495982f1cdf` |
| PyPTO | `nalinaly/pypto`，`e891351e305e1b8ac70ffb9ffe60a243fc0420dc` |
| Simpler | `nalinaly/simpler`，`57e499238fd4ee3548947ad860be456aaeeb16b5` |
| PTOAS | `nalinaly/PTOAS`，`914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，CLI 0.69 |
| PTO-ISA | `hw-native-sys/pto-isa`，`327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` |
| LLVM / MLIR | `vpto-dev/llvm-project`，`fa2fd1f75d742fec91ecf40d98c9c7961647ec42`，加第 4 节补丁 |
| PTOAS 的 CMake 工具依赖 | `gitcode.com/cann/cmake`，`a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70` |
| 硬件 | Linux aarch64 / Ascend A3，单设备 24 AIC + 48 AIV |
| CANN | 9.2.0-beta.2 |
| Python / Torch / torch_npu | 3.10.19 / 2.10.0 / 2.10.0.post2 |
| SDK Host 编译器 / CMake | GCC 15.2.0 / CMake 4.4.2 |
| 权重输入（二选一） | [真实权重：DeepSeek-V4-Flash-0731-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8) 第 4 层；或按第 3.1 节生成随机 checkpoint |

需要已安装的 A3 驱动与 CANN、GCC 15、Python 开发头文件、git、make。
实测驱动为 26.0.rc1、ascendhal 7.35.23。模型目录需包含原始 ModelSlim 权重索引、
`config.json` 和第 4 层对应的 safetensors 分片。无须原作者的 KV 文件或实验 bundle。

本流程从新 Python 环境和源码构建开始；驱动、CANN、GCC 需预先准备。
模型权重可自行下载，也可用第 3.1 节的随机 checkpoint 做功能验证。

## 2. 浅拉取源码

在同一个 Bash 会话执行；选择新的空工作目录。目录名用于匹配已提交的 SDK 清单，
其中 `.worktrees` 可以是普通 clone，不要求使用 Git worktree。

```bash
set -eo pipefail
export CSA_WORKSPACE=/your/workspace/csa-torch-repro
mkdir -p "$CSA_WORKSPACE/.worktrees" "$CSA_WORKSPACE/deps" "$CSA_WORKSPACE/results"
cd "$CSA_WORKSPACE"

fetch_pinned() {
    local url=$1 revision=$2 destination=$3
    test ! -e "$destination" || { printf '目录已存在：%s\n' "$destination" >&2; return 1; }
    git init -q "$destination" || return
    git -C "$destination" remote add origin "$url" || return
    git -C "$destination" fetch --depth=1 origin "$revision" || return
    git -C "$destination" checkout --detach FETCH_HEAD || return
}

fetch_pinned https://github.com/nalinaly/vllm-ascend.git \
  649d6214de8f247883520d6f9df8762b7ee7876e "$CSA_WORKSPACE/csa-source"
fetch_pinned https://github.com/nalinaly/pypto.git \
  e891351e305e1b8ac70ffb9ffe60a243fc0420dc \
  "$CSA_WORKSPACE/.worktrees/pypto-csa-mpmd-upstream-20261021"
fetch_pinned https://github.com/nalinaly/simpler.git \
  57e499238fd4ee3548947ad860be456aaeeb16b5 \
  "$CSA_WORKSPACE/.worktrees/simpler-csa-core-args-256-20261016"
fetch_pinned https://github.com/nalinaly/PTOAS.git \
  914863ed3d0eb3e92a4433a4a706a86cabf4c91f \
  "$CSA_WORKSPACE/.worktrees/ptoas-csa-mpmd-upstream-20261021"

export CSA_REPO="$CSA_WORKSPACE/csa-source"
export CSA_PYPTO="$CSA_WORKSPACE/.worktrees/pypto-csa-mpmd-upstream-20261021"
export CSA_SIMPLER="$CSA_WORKSPACE/.worktrees/simpler-csa-core-args-256-20261016"
export CSA_PTOAS="$CSA_WORKSPACE/.worktrees/ptoas-csa-mpmd-upstream-20261021"

git -C "$CSA_REPO" fetch --depth=1 origin 84751e34cb1f27eff9e4d3fdf03f8495982f1cdf
git -C "$CSA_PYPTO" config submodule.simpler.url https://github.com/nalinaly/simpler.git
git -C "$CSA_PYPTO" submodule update --init --recursive --depth=1
mkdir -p "$CSA_SIMPLER/build"
fetch_pinned https://github.com/hw-native-sys/pto-isa.git \
  327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d "$CSA_SIMPLER/build/pto-isa"

# PTOAS 构建所需的 CMake 工具；GitCode 直连，不修改全局代理配置。
(
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    git init -q "$CSA_WORKSPACE/deps/cann-cmake"
    git -C "$CSA_WORKSPACE/deps/cann-cmake" remote add origin https://gitcode.com/cann/cmake.git
    git -C "$CSA_WORKSPACE/deps/cann-cmake" -c http.proxy= fetch --depth=1 origin \
      a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70
    git -C "$CSA_WORKSPACE/deps/cann-cmake" checkout --detach FETCH_HEAD
)
```

不执行算子源码仓的 submodule update 或 pip install。PyPTO 的 runtime 子模块须为
上述 Simpler 提交；PTO-ISA 使用 Simpler 的 `pto_isa.pin`，不要另外切换版本。

## 3. 独立 Python 环境

将路径替换为本机安装位置。CANN 使用自己的 `set_env.sh`；无需其他推理框架环境脚本。

```bash
cat > "$CSA_WORKSPACE/repro-local.sh" <<'BASH'
export CSA_CANN_ROOT=/path/to/cann-9.2.0-beta.2
export CSA_GCC15=/path/to/gcc-15.2.0
export CSA_MODEL=/path/to/DeepSeek-V4-Flash-0731-w8a8
export CSA_LLVM_JOBS=8
export CSA_PTOAS_JOBS=8
BASH
source "$CSA_WORKSPACE/repro-local.sh"
python3.10 -m venv "$CSA_WORKSPACE/.venv"
source "$CSA_WORKSPACE/.venv/bin/activate"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export NO_PROXY="pypi.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

python -m pip install \
  torch==2.10.0 torch-npu==2.10.0.post2 numpy==2.2.6 safetensors==0.8.0 \
  cloudpickle==3.1.1 pybind11==3.1.0 pyyaml==6.0.3 psutil==6.0.0 tabulate==0.9.0 \
  attrs==24.2.0 decorator==5.1.1 scipy==1.13.1 protobuf==7.36.2 \
  cmake==4.4.2 ninja==1.13.0 nanobind==2.15.0 scikit-build-core==1.0.3 \
  setuptools==79.0.1 setuptools-scm==10.2.3 wheel==0.48.0
```

`attrs/decorator/scipy` 等是 CANN Python 编译组件的依赖。这里无需 Transformers、
Triton、模型插件或 Native OPP 包。第一次设备运行仍会编译 PTO 内核。

### 3.1 没有模型权重：生成随机 checkpoint（可上卡）

只想跑通编译、单 SPMD/MPMD 对照、图重放和泳道，选择此路径即可；有真实权重则跳过。
完成上面的 Python 安装后调用 [生成脚本](csa_torch_repro/generate_random_checkpoint.py)，**不下载或读取任何模型文件，不占用 NPU**。
它生成固定 seed 的 28 个张量及最小配置，约 191 MiB；INT8 权重使用正 scale 和零 offset，
矩阵转置及 NZ 转换仍由原测试入口完成，两种实现使用同一套数据。

```bash
# 固定算子提交不含新脚本；仅从本手册所在分支提取脚本，不切换算子版本。
git -C "$CSA_REPO" fetch --depth=1 origin dsv4-flash-pto-v0.25.1rc1
git -C "$CSA_REPO" show FETCH_HEAD:tests/pypto_test/csa_torch_repro/generate_random_checkpoint.py \
  > "$CSA_WORKSPACE/generate_random_checkpoint.py"
python "$CSA_WORKSPACE/generate_random_checkpoint.py" \
  --output "$CSA_WORKSPACE/random-csa-weights"
```

然后将 `repro-local.sh` 中的 `CSA_MODEL` 那一行**替换**为：

```bash
export CSA_MODEL="$CSA_WORKSPACE/random-csa-weights"
```

继续执行第 4～6 节，第 6 节运行标签建议用 `randomfirst`，避免与真实权重结果混淆。
无需修改算子或测试入口；`random-csa-weights/SYNTHETIC_WEIGHTS.json` 标记数据来源。
生成目录已存在时脚本会报错，不会覆盖已有文件；需要重跑算子时直接复用该目录。

**验收边界：**只检查两种实现对随机数据的一致性及保护区，不证明模型精度，
也不保证覆盖真实数据的所有分支。Top-K、压缩和访存具有数据相关性，
随机权重下的耗时不能当作真实权重性能；这里的最小配置也不能用于整模型推理。

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

## 5. 环境入口与正式算子准备

```bash
cat > "$CSA_WORKSPACE/env-csa-torch.sh" <<'BASH'
export CSA_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CSA_WORKSPACE/repro-local.sh"
unset PYTHONPATH ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH
source "$CSA_CANN_ROOT/set_env.sh"
source "$CSA_WORKSPACE/.venv/bin/activate"
export PATH="$CSA_GCC15/bin:$PATH"
export LD_LIBRARY_PATH="$CSA_GCC15/lib64:$CSA_WORKSPACE/deps/llvm-build/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export CSA_REPO="$CSA_WORKSPACE/csa-source"
export CSA_TESTS="$CSA_REPO/tests/pypto_test"
export CSA_CASES="$CSA_TESTS/csa_mpmd_tnd_20261021"
BASH
source "$CSA_WORKSPACE/env-csa-torch.sh"
python "$CSA_CASES/prepare_experiment.py" repro_formal
```

准备器只从 Git 提取冻结的 PTO 源码并复制正式 MPMD，实现不依赖未提交实验目录。
`csa_torch_repro/bootstrap.py` 按清单绑定同一套 PyPTO、Simpler 和 PTOAS，
以源码命名空间加载 PTO 包；禁止导入 vLLM 和 Native 扩展，不执行推理插件初始化。

128K/B16 的固定场景包含 16 个活动请求、17 个请求槽、55 个有效 token、64 个 token
容量。真实 query 长度为 `[6,1,4,2,5,3,6,1,4,2,5,3,6,1,4,2]`，历史长度保持原清单；
最后一个请求 L=0、slots=-1，承接 9 行 padding。

`csa_torch_repro/inputs.py` 直接读取第 4 层权重，显式处理 INT8 矩阵方向、BF16 norm
和四张 NZ 根权重；按固定 seed 生成历史 cache/state。Indexer 保持每页 4096 字节 key
加 64 字节 FP16 scale，主/Indexer state 保留各自物理页步长及保护区。
这些准备均在计时区间外；算子内部的数据相关计划仍由原 PTO 实现计算。

## 6. 执行 PyTorch 单卡验证

先保存一次运行脚本。正式版对照与第 7 节的三方对照二选一即可。

```bash
cat > "$CSA_WORKSPACE/run-csa-torch.sh" <<'BASH'
#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env-csa-torch.sh"
case_name=${1:-h131072_b16}
experiment=${2:-repro_formal}
label=${3:?请指定新的输出标签}
shift 3
control=
if [[ $# -gt 0 && "$1" != --device ]]; then control=$1; shift; fi
if [[ ${1:-} == --device ]]; then
    [[ $# -eq 2 && "$2" == "${TASK_DEVICE:?}" ]] || exit 2
    shift 2
fi
if (( $# )); then exit 2; fi
: "${TASK_DEVICE:?由设备调度服务分配单张卡}"
out="$CSA_WORKSPACE/results/${experiment}_${case_name}_${label}"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH"
extra=()
if [[ -n "$control" ]]; then extra+=(--control-mpmd "$CSA_CASES/experiments/$control/operator"); fi
exec python "$CSA_TESTS/csa_torch_repro/bootstrap.py" --workspace-root "$CSA_WORKSPACE" \
  "$CSA_TESTS/csa_torch_repro/run.py" --case "$case_name" --model "$CSA_MODEL" \
  --frozen "$CSA_CASES/sources/84751e34" --candidate "$CSA_CASES/experiments/$experiment/operator" \
  --output "$out" "${extra[@]}"
BASH

printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-csa-torch.sh" h131072_b16 repro_formal first
task-submit --device auto --max-time 1500 "$CSA_COMMAND"
```

`task-submit` 是本机共享设备队列，不是 PyPTO 的一部分。其他机器通过自己的设备调度
分配单张卡，并在调用运行脚本前设置 `TASK_DEVICE`；不要与已有可见卡映射产生冲突。
本机用 `task-submit --status <task-id>`、`--log <task-id>` 查询终态。

检查输出目录的 `result.json`：

- 任务退出码 0、`stage=complete`、`passed=true`。
- `checks` 中 eager、graph、post_l1 的数值与保护区均通过。
- output、Top-K、scores、cache/state 分别遵守原场景的明确容差；整数/索引和保护区精确一致。
- `dependency_check` 显示没有导入 vLLM、没有注册 `_C_ascend` 自定义算子。
- `statistics_us` 保存每侧的 min/mean/max 和全部 5 个样本；`l1_00_*/swimlane.json` 可查看一次执行。

完整 CSA 范围为 HC_pre → input RMSNorm → CSA → HC_post，计时取首个 incore 开始到
最后一个 incore 结束，包含内部等待。warmup 3 次后交替采样 5 次，不累加重叠 task，
不把首次编译、输入恢复、CPU 校验或 D2H 时间计入 L1。

## 7. 可选：v25 三方对照与其他固定档位

```bash
python "$CSA_CASES/prepare_experiment.py" v17_upstream_fixes
python "$CSA_CASES/prepare_priority_fusion.py" v18_priority_fusion
python "$CSA_CASES/prepare_query_group.py" v19_query_group v18_priority_fusion
python "$CSA_CASES/prepare_query_score_ready.py" v21_query_release_priority v19_query_group --release-query-priority
python "$CSA_CASES/prepare_front_q_priority.py" v22_front_q_priority v21_query_release_priority
python "$CSA_CASES/prepare_query_weights.py" v23_query_weights v22_front_q_priority
python "$CSA_CASES/prepare_score_live_half.py" v24_score_live_half v23_query_weights
python "$CSA_CASES/prepare_score_leaf_rotation.py" v25_score_leaf_rotation v24_score_live_half

printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-csa-torch.sh" \
  h131072_b16 v25_score_leaf_rotation first repro_formal
task-submit --device auto --max-time 1500 "$CSA_COMMAND"
```

该命令同进程比较冻结 SPMD、正式 MPMD 控制组和 v25；正式 MPMD 对应结果中的
`mpmd_control`，v25 对应 `mpmd`。每个实验只生成一次，排队和执行期间不修改其源码。

其他固定场景为 `h131072_b1`、`h131072_b4`、`h131072_b8`、`h131072_b24`、
`h131072_b28`、`h8192_b16`、`h8192_b32`，替换运行命令的第一个参数即可。
保持 [cases.json](csa_mpmd_tnd_20261021/cases.json) 的输入、seed、padding 与容差。

## 8. 独立入口验证结果

已在未安装 vLLM/vLLM-Ascend 的新 Python venv 中完成 128K/B16 验证，
使用上述独立入口、正式单层权重及固定 TND 场景。任务
`task_20261022_022812_398481424355` 退出码为 0。

| 实现 | min（μs） | mean（μs） | max（μs） |
| --- | --- | --- | --- |
| 冻结 TND SPMD | 700.58 | 710.724 | 715.24 |
| 正式 TND MPMD | 891.02 | 900.476 | 912.14 |
| v25 TND MPMD | 839.04 | 843.088 | 852.08 |

三种实现的 eager、NPUGraph replay、post-L1 共 9 组检查全部通过；每组 95 或 96 项
保护检查通过，9 类输出/cache/state 对冻结 SPMD 逐 bit 一致。15 个 L1 窗口均已导出。
运行结束的依赖检查为 `vllm_imported=false`、`native_custom_ops_registered=false`。

新入口的输入准备完全由 PyTorch 完成，计时结果只按本次同进程的配对样本比较。
接收方应保存本机 `result.json`，要求数值及保护检查通过；绝对时间受设备频率和负载影响，
不要求精确等于表中的微秒值。独立环境实测覆盖 128K/B16，其余固定档位可按需执行。
