# DeepSeek V4 Flash CSA PyTorch 单卡复现手册（2026-10-22）

本文包含 **vLLM-Ascend、vLLM、PyPTO、Simpler、PTOAS、PTO-ISA** 的完整源码来源、
构建顺序与 **PyTorch 单卡数值、图重放、L1 性能验证**。命令针对 Linux aarch64 / Ascend A3，使用自己的工作目录，
不需要原机器的 `.worktrees`、编译缓存、editable 安装或未提交文件。

主复现对象是原生 TND 多 task CSA：
`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/`。
同进程参考是冻结在 `84751e34` 的 TND 单 SPMD。两个入口均包括
HC_pre → input RMSNorm → CSA → HC_post。本文也给出已提交的 v25 候选的生成方法。

复现验收为 PyTorch 单卡。单卡脚本借用 vLLM/vLLM-Ascend 的配置、
单层权重适配及 Native metadata builder，所以保留这两个源码包的安装。
输入和历史 cache/state 由固定场景在本机生成；无须取得原作者的缓存文件。
通过 PyTorch/torch_npu 直接调用 PTO 算子并检查 NPUGraph replay，
superkernel 和 npugraph_ex static kernel 均关闭。PyPTO/PTOAS/CCEC 编译是必要构建步骤。

## 1. 固定源码版本

以下完整 SHA 是复现依据；分支名只用于查找。不要以 main 或调试分支的最新提交替换。

| 仓库 | 获取地址 | 固定 commit |
| --- | --- | --- |
| vLLM-Ascend | <https://github.com/nalinaly/vllm-ascend>，`dsv4-flash-pto-v0.25.1rc1` | `6323cc35816c4e0e73dcd9b488cc26047ada33da` |
| vLLM | <https://github.com/vllm-project/vllm> | `752a3a504485790a2e8491cacbb35c137339ad34` |
| PyPTO | <https://github.com/nalinaly/pypto>，`feat/csa-mpmd-upstream-fixes-20261021` | `e891351e305e1b8ac70ffb9ffe60a243fc0420dc` |
| Simpler | <https://github.com/nalinaly/simpler>，`dsv4-csa-spmd-repro-20261019` | `57e499238fd4ee3548947ad860be456aaeeb16b5` |
| PTOAS | <https://github.com/nalinaly/PTOAS>，`feat/csa-mpmd-upstream-fixes-20261021` | `914863ed3d0eb3e92a4433a4a706a86cabf4c91f` |
| PTO-ISA | <https://github.com/hw-native-sys/pto-isa> | `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` |
| LLVM / MLIR | <https://github.com/vpto-dev/llvm-project> | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42` |
| CANN CMake | <https://gitcode.com/cann/cmake>，`master-055` | `a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70` |
| CATLASS | <https://gitcode.com/cann/catlass> | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |

vLLM-Ascend 的 SHA 固定本文编写前的计算代码和测试代码；后续文档提交不改变这个分母。
Python 包显示的 vllm-ascend 版本可能仍带早期 editable 安装时的 Git 后缀，应核对实际源码
路径和上述 Git SHA。PTOAS 显示 0.69，但该字符串也不能代替源码提交。

当前 PyPTO 的 `runtime` gitlink 已指向上述 Simpler 提交，`.gitmodules` 的 URL
仍指向上游。下文在克隆后用仓库本地配置指定 `nalinaly/simpler`，不改源码，不执行
`submodule update --remote`。PTO-ISA 由 Simpler 的 `pto_isa.pin` 决定，设置一个
任意的 `PTO_ISA_ROOT` 环境变量不能替换这个 pin。

## 2. 硬件、系统软件和权重

| 项目 | 当前实测组合 |
| --- | --- |
| Host / NPU | Linux aarch64；A3，每设备 24 AIC + 48 AIV |
| 驱动 | 26.0.rc1；`ascendhal_version=7.35.23` |
| Python | 3.10.19 |
| Torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| Transformers / NumPy | 5.14.1 / 2.2.6 |
| Native / PTO 编译和运行 CANN | 统一使用 9.2.0-beta.2 |
| ATB | 9.0.0 包，`--cxx_abi=1` |
| Host 编译器 | PyPTO/PTOAS：GCC 15.2.0；Native 构建：GCC 10.3.1 |
| CMake | PyPTO/PTOAS：4.4.2；Native 自定义算子构建：3.31.10 |
| 模型 | 正式 `DeepSeek-V4-Flash-0731-w8a8`，包含 `quant_model_weights.safetensors.index.json` |
| 单层口径 | 层索引 4（`layers.4`）、TP=1、NZ mode=2、DSpark 出5验6，不启用 EPLB |

先安装可用的 A3 驱动/固件，以及表中的 CANN、ATB、Python 和编译器。模型权重、
驱动和 CANN 安装包不在 Git 仓库中，需要从各自合法分发渠道取得。
本次不声称可以由本机记录还原未保存的驱动安装包或历史 Native 二进制的逐字节构建。
新机器会从固定源码重新构建；需通过下文正确性检查，并在本机重取性能结果。

系统构建工具还需要 `git`、`make`、`patch`、`curl`、`pkg-config` 和 Python 开发头文件。
其中 `patch` 用于 Native 的 abseil/protobuf 补丁，缺少时构建会直接失败；可由系统包管理器
安装到构建环境，或自行构建 GNU patch 并把其 `bin` 加入 `PATH`。

源码中 Triton Ascend 的依赖声明为 3.2.1，当前环境实际是 3.2.2；FastAPI 的两仓
依赖声明也存在冲突。为重现实际可运行组合，下文安装附录中的实测版本，再用
`--no-deps` 安装两个源码仓，避免 resolver 悄悄改动 Torch、Triton 或模型依赖。
这不是将 `pip check` 全绿作为本组合的既有结论。
固定 vLLM 源码的 build-system 声明 Torch 2.11，而此 Ascend 组合要求 Torch 2.10；
因此安装 empty backend 时也须保留 `--no-build-isolation`，不让临时构建环境改用 2.11。

本机的 NPU 执行通过 `task-submit` 队列；安装、编译和 CPU 检查不需要占卡。
`task-submit` 是这台共享机器的资源调度服务，不是 vLLM/PyPTO 自带的命令。
其他机器须由自己的调度服务分配独占设备，并将该进程可见的设备序号写入 `TASK_DEVICE`，
再调用第 9 节的运行脚本；不要同时设置相互矛盾的可见卡映射和物理设备号。

## 3. 获取源码：全部浅拉取

以下命令在同一个 Bash 会话执行。选择一个新的空工作目录，完整路径中不要包含以 `.`
开头的隐藏目录，以满足 Native 算子打包工具的路径要求。

```bash
set -eo pipefail
export CSA_WORKSPACE=/your/workspace/csa-repro-20261021
mkdir -p "$CSA_WORKSPACE/.worktrees" "$CSA_WORKSPACE/deps" "$CSA_WORKSPACE/results"
cd "$CSA_WORKSPACE"

fetch_pinned() {
    local url=$1 revision=$2 destination=$3
    test ! -e "$destination" || { printf '目录已存在：%s\n' "$destination" >&2; return 1; }
    git init -q "$destination" || return
    git -C "$destination" remote add origin "$url" || return
    git -C "$destination" fetch --depth=1 origin "$revision" || return
    git -C "$destination" checkout --detach FETCH_HEAD || return
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}

fetch_pinned https://github.com/nalinaly/vllm-ascend.git \
  6323cc35816c4e0e73dcd9b488cc26047ada33da "$CSA_WORKSPACE/vllm-ascend"
fetch_pinned https://github.com/vllm-project/vllm.git \
  752a3a504485790a2e8491cacbb35c137339ad34 "$CSA_WORKSPACE/vllm"
fetch_pinned https://github.com/nalinaly/pypto.git \
  e891351e305e1b8ac70ffb9ffe60a243fc0420dc \
  "$CSA_WORKSPACE/.worktrees/pypto-csa-mpmd-upstream-20261021"
fetch_pinned https://github.com/nalinaly/simpler.git \
  57e499238fd4ee3548947ad860be456aaeeb16b5 \
  "$CSA_WORKSPACE/.worktrees/simpler-csa-core-args-256-20261016"
fetch_pinned https://github.com/nalinaly/PTOAS.git \
  914863ed3d0eb3e92a4433a4a706a86cabf4c91f \
  "$CSA_WORKSPACE/.worktrees/ptoas-csa-mpmd-upstream-20261021"

export CSA_REPO="$CSA_WORKSPACE/vllm-ascend"
export CSA_PYPTO="$CSA_WORKSPACE/.worktrees/pypto-csa-mpmd-upstream-20261021"
export CSA_SIMPLER="$CSA_WORKSPACE/.worktrees/simpler-csa-core-args-256-20261016"
export CSA_PTOAS="$CSA_WORKSPACE/.worktrees/ptoas-csa-mpmd-upstream-20261021"

git -C "$CSA_PYPTO" config submodule.simpler.url https://github.com/nalinaly/simpler.git
git -C "$CSA_PYPTO" submodule update --init --recursive --depth=1
test "$(git -C "$CSA_PYPTO/runtime" rev-parse HEAD)" = \
  57e499238fd4ee3548947ad860be456aaeeb16b5
mkdir -p "$CSA_SIMPLER/build"
fetch_pinned https://github.com/hw-native-sys/pto-isa.git \
  327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d "$CSA_SIMPLER/build/pto-isa"

# 固定测试会 git archive 此参考提交；浅克隆必须额外取回这个 Git 对象。
git -C "$CSA_REPO" fetch --depth=1 origin 84751e34cb1f27eff9e4d3fdf03f8495982f1cdf

# GitCode 不走代理；不修改全局 Git 配置。
(
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    git -C "$CSA_REPO" -c http.proxy= submodule update --init --recursive --depth=1
    git init -q "$CSA_WORKSPACE/deps/cann-cmake"
    git -C "$CSA_WORKSPACE/deps/cann-cmake" remote add origin https://gitcode.com/cann/cmake.git
    git -C "$CSA_WORKSPACE/deps/cann-cmake" -c http.proxy= fetch --depth=1 origin \
      a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70
    git -C "$CSA_WORKSPACE/deps/cann-cmake" checkout --detach FETCH_HEAD
)
```

每条命令成功后再执行下一步；网络失败时修复访问后重试失败项，不把浅克隆失败理解为
必须下载整仓历史。以上 `.worktrees` 目录名用于匹配仓内公共清单，可以是普通 clone，
无需先创建 Git worktree 主仓。

## 4. 本机路径配置与 Python 环境

先把以下配置中的路径替换为本机实际安装位置。GCC 两组路径分别对应表中的版本。

```bash
cat > "$CSA_WORKSPACE/repro-local.sh" <<'BASH'
export CSA_CANN_ROOT=/path/to/cann-9.2.0-beta.2
export CSA_ATB=/path/to/atb-package-9.0.0/atb
export CSA_GCC15=/path/to/gcc-15.2.0
export CSA_NATIVE_CC=/path/to/gcc-10.3.1/bin/gcc
export CSA_NATIVE_CXX=/path/to/gcc-10.3.1/bin/g++
export CSA_MODEL=/path/to/DeepSeek-V4-Flash-0731-w8a8
export CSA_SOC=ascend910_9392
export CSA_LLVM_JOBS=8
export CSA_NATIVE_JOBS=8
export CSA_PTOAS_JOBS=8
BASH
source "$CSA_WORKSPACE/repro-local.sh"

# 若系统没有 patch，可安装到本工作目录，无需修改系统软件。
if ! command -v patch >/dev/null; then
    curl --noproxy '*' -fL --retry 2 --max-time 300 \
      https://ftp.gnu.org/gnu/patch/patch-2.7.6.tar.xz \
      -o "$CSA_WORKSPACE/deps/patch-2.7.6.tar.xz"
    tar -xJf "$CSA_WORKSPACE/deps/patch-2.7.6.tar.xz" -C "$CSA_WORKSPACE/deps"
    (
        cd "$CSA_WORKSPACE/deps/patch-2.7.6"
        ./configure --prefix="$CSA_WORKSPACE/host-tools" CC="$CSA_NATIVE_CC"
        make -j2
        make install
    )
fi
export PATH="$CSA_WORKSPACE/host-tools/bin:$PATH"

python3.10 -m venv "$CSA_WORKSPACE/.venv"
source "$CSA_WORKSPACE/.venv/bin/activate"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2
# 本机 PyPI 经代理返回 403；这几个下载域直连，GitHub 可继续使用原代理。
export NO_PROXY="pypi.org,files.pythonhosted.org,pypi.tuna.tsinghua.edu.cn,mirrors.huaweicloud.com,gitcode.com${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

执行文末附录的代码块，将 `requirements-csa-observed.txt` 写入工作目录，然后安装。
锁表来自本机已安装包的依赖闭包，排除了源码安装的 vLLM/vLLM-Ascend 和单独安装的
Triton；不包含令牌、私有索引地址、editable 路径或编译产物。

```bash
python -m pip install uv==0.9.28
uv pip install --python "$CSA_WORKSPACE/.venv/bin/python" \
  --index-url "$PIP_INDEX_URL" --no-deps --link-mode copy \
  -r "$CSA_WORKSPACE/requirements-csa-observed.txt"
uv pip install --python "$CSA_WORKSPACE/.venv/bin/python" \
  --index-url "$PIP_INDEX_URL" --no-deps --link-mode copy triton==3.5.0
uv pip install --python "$CSA_WORKSPACE/.venv/bin/python" --no-deps --link-mode copy \
  --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.2
# 固定源码声明但当前单层环境未安装的可选语法缓存依赖；补齐完整 vLLM 导入路径。
python -m pip install --no-deps diskcache==5.6.3

env -u VLLM_VERSION_OVERRIDE VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1 \
  python -m pip install --no-build-isolation --no-deps \
  -e "$CSA_WORKSPACE/vllm"
SOC_VERSION="$CSA_SOC" COMPILE_CUSTOM_KERNELS=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1rc1 \
  python -m pip install --no-build-isolation --no-deps -e "$CSA_REPO"

# Native 构建使用独立 CMake 3.31，避免改变 PTO 构建环境中的 CMake 4.4。
python -m pip install --no-deps --target "$CSA_WORKSPACE/native-build-tools" cmake==3.31.10
```

若某个指定版本无法从公共索引取得，应取得该版本的 wheel/源码再安装，不自动升级到最新。
这里用 uv 0.9.28 安装同一份锁表，可复用第三方发布包的下载缓存；PyPTO、PTOAS、Simpler
和 Native 扩展仍从固定源码重新构建。Triton Ascend 3.2.2 使用单独的 Ascend 索引。
上述 `COMPILE_CUSTOM_KERNELS=0` 只登记 Python 插件；下一节必须实际构建 Native 扩展。
浅拉取不包含 release tag，安装时显式指定 Python 包版本标签，避免 setuptools-scm
生成 `0.1.dev0`；源码身份始终以上述 SHA 为准，不据此声称是官方发行二进制。
vLLM 要用 `SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1`：它生成的 `vllm.__version__`
为 `0.25.1`，empty backend 的发行包标签则为 `0.25.1+empty`，与原环境一致。
不能改为 `VLLM_VERSION_OVERRIDE=0.25.1+empty`，否则运行时版本比较失败，
会误走另一版 FLA 导入路径并在模型初始化时报错。

## 5. 构建 vLLM-Ascend 的 Native 依赖

从固定 vLLM-Ascend 源码构建本次选定的 14 个 ACLNN 自定义算子及 C++ 扩展，
具体算子清单见下方 `--ops` 参数；单卡 PTO 测试实际调用其中的 Native metadata 生成接口。
Native 与 PTO 使用同一套 CANN 9.2，不要求另外安装 CANN 9.0。
以下在子 shell 中运行，Native 所需的 GCC/CMake 选择不会影响后续 PTO 构建。
这里编译的是本仓 `csrc/` 的自定义算子，并非重编 CANN 自带的算子库。
CANN 9.2 已提供 Compressor、QuantLightningIndexer 等内置算子，但本仓当前调用的
`aclnnCompressorMetadata`、`aclnnVllmQuantLightningIndexerMetadata` 等自定义接口并未内置；
仅安装 CANN、只构建 Python/C++ 绑定，不能补齐这些 metadata kernel。

```bash
(
    set -eo pipefail
    source "$CSA_CANN_ROOT/set_env.sh"
    source "$CSA_ATB/set_env.sh" --cxx_abi=1
    export CC="$CSA_NATIVE_CC" CXX="$CSA_NATIVE_CXX"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CSA_NATIVE_JOBS:-2}" MAX_JOBS="${CSA_NATIVE_JOBS:-2}"
    export PATH="$CSA_WORKSPACE/native-build-tools/cmake/data/bin:$CSA_WORKSPACE/.venv/bin:$(dirname "$CC"):$PATH"
    export CPATH="$CSA_REPO/csrc/third_party/catlass/include${CPATH:+:$CPATH}"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    # 下载位置遵循 --cann_3rd_lib_path，但本仓 CPack 配置仍查找 third_party/makeself。
    if [[ ! -e "$CSA_REPO/csrc/third_party/makeself" && ! -L "$CSA_REPO/csrc/third_party/makeself" ]]; then
        ln -s "$CSA_WORKSPACE/deps/makeself" "$CSA_REPO/csrc/third_party/makeself"
    fi
    cd "$CSA_REPO/csrc"
    bash build.sh --pkg --soc=ascend910_93 -j"${CSA_NATIVE_JOBS:-2}" \
      --cann_3rd_lib_path="$CSA_WORKSPACE/deps" \
      --ops='compressor;compressor_metadata;vllm_quant_lightning_indexer;vllm_quant_lightning_indexer_metadata;sparse_attn_sharedkv;sparse_attn_sharedkv_metadata;rms_norm_dynamic_quant;inplace_partial_rotary_mul;scatter_nd_update_v2;hc_pre;hc_post;moe_gating_top_k_hash;moe_gating_top_k;dequant_swiglu_quant'
    shopt -s nullglob
    installers=("$CSA_REPO"/csrc/build/cann-ops-transformer*.run)
    test "${#installers[@]}" = 1
    bash "${installers[0]}" --install-path="$CSA_REPO/vllm_ascend/_cann_ops_custom"

    cmake -S "$CSA_REPO" -B "$CSA_REPO/build-native-repro" \
      -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="$CSA_SOC" \
      -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
      -DASCEND_HOME_PATH="$CSA_CANN_ROOT" \
      -DPYTHON_EXECUTABLE="$(command -v python)" \
      -DPYTHON_INCLUDE_PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
      -DTORCH_NPU_PATH="$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch_npu").origin).parent)')" \
      -DCMAKE_INSTALL_PREFIX="$CSA_REPO/vllm_ascend"
    cmake --build "$CSA_REPO/build-native-repro" --parallel "${CSA_NATIVE_JOBS:-2}"
    cmake --install "$CSA_REPO/build-native-repro"
)
```

上述 14 项是本次单卡验证实际使用的 Native 安装组合，不声称是裁剪后的最小依赖集。
单卡脚本使用 Native metadata 与布局适配接口；这一步是准备运行依赖，不是启动整模型。
从空目录一次构建该清单。

## 6. 构建四个 PTO 依赖

### 6.1 加载 CANN 9.2，准备 LLVM / PTOAS

继续使用前面同一个 shell；Native 的 GCC/CMake 已被限制在子 shell 内。

```bash
source "$CSA_CANN_ROOT/set_env.sh"
source "$CSA_ATB/set_env.sh" --cxx_abi=1
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

### 6.2 Simpler 与 PTO-ISA

```bash
python -m venv --system-site-packages "$CSA_SIMPLER/.venv"
"$CSA_SIMPLER/.venv/bin/python" -m pip install \
  nanobind==2.15.0 scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0 cloudpickle==3.1.2
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

### 6.3 PyPTO

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

## 7. 建立可供队列任务加载的环境入口

公共工具链清单使用工作目录下的相对路径，前三节的目录布局即可直接匹配它。
不要 source 原作者机器上的 `env-dsv4-0251rc1.sh`。

```bash
cat > "$CSA_WORKSPACE/env-csa-repro.sh" <<'BASH'
export CSA_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CSA_WORKSPACE/repro-local.sh"
unset PYTHONPATH
source "$CSA_CANN_ROOT/set_env.sh"
source "$CSA_ATB/set_env.sh" --cxx_abi=1
source "$CSA_WORKSPACE/.venv/bin/activate"
export CSA_REPO="$CSA_WORKSPACE/vllm-ascend"
export CSA_TESTS="$CSA_REPO/tests/pypto_test"
export PYTHONPATH="$CSA_WORKSPACE/vllm:$CSA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CC="$CSA_GCC15/bin/gcc-15" CXX="$CSA_GCC15/bin/g++-15"
export PATH="$CSA_GCC15/bin:$PATH"
export LD_LIBRARY_PATH="$CSA_GCC15/lib64:$CSA_WORKSPACE/deps/llvm-build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_CUSTOM_OPP_PATH="$CSA_REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=spmd_tnd
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
source "$CSA_TESTS/csa_shared_toolchain/env.sh"
BASH
source "$CSA_WORKSPACE/env-csa-repro.sh"

python "$CSA_TESTS/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --workspace-root "$CSA_WORKSPACE" --vllm-root "$CSA_WORKSPACE/vllm" \
  "$CSA_TESTS/csa_shared_toolchain/show_versions.py" --output "$CSA_WORKSPACE/results/toolchain.json"
```

`toolchain.json` 应显示父子进程来源一致，PyPTO 包和扩展来自指定源码树，Simpler 的扩展
来自其自己的 `.venv`，PTOAS 来自对应 `build/bin/ptoas`。该检查不初始化 NPU。

## 8. 准备正式算子与冻结场景

```bash
export CSA_CASES="$CSA_TESTS/csa_mpmd_tnd_20261021"
python "$CSA_CASES/prepare_experiment.py" repro_formal

python - <<'PY'
import json, os
from pathlib import Path
workspace = Path(os.environ['CSA_WORKSPACE'])
source = Path(os.environ['CSA_CASES']) / 'cases.json'
data = json.loads(source.read_text())
data['model'] = os.environ['CSA_MODEL']
(workspace / 'cases.local.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
PY
```

这里只替换权重所在路径，权重内容须仍是同一正式模型。不能修改 `cases` 中的 query 长度、
history、padding、seed 或正确性容差。`prepare_experiment.py` 会从已获取的 `84751e34`
生成参考源码和测试 helper，再冻结当前正式 MPMD 源码；不依赖原机器未上传的实验目录。

128K/B16 是 **16 个活动请求、17 个请求槽、55 个有效 token、64 个 token 容量**。
真实 query 长度为 `[6,1,4,2,5,3,6,1,4,2,5,3,6,1,4,2]`，最后一个 dummy 请求的
L=0、slots=-1，吸收 9 行 padding。它不是统一 S6 的 B×6 场景。

先做不占卡的完整后端编译。输出目录须不存在：

```bash
export PYPTO_PROG_BUILD_DIR="$CSA_WORKSPACE/results/cpu-b16/build_output"
export PYPTO_CACHE_DIR="$CSA_WORKSPACE/results/cpu-b16/cache"
python "$CSA_TESTS/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --workspace-root "$CSA_WORKSPACE" --vllm-root "$CSA_WORKSPACE/vllm" \
  "$CSA_CASES/experiments/repro_formal/run_case.py" \
  --cpu --compile --case h131072_b16 --cases "$CSA_WORKSPACE/cases.local.json" \
  --candidate "$CSA_CASES/experiments/repro_formal/operator" \
  --frozen "$CSA_CASES/sources/84751e34" --output "$CSA_WORKSPACE/results/cpu-b16"
```

通过标志为 `stage=cpu_backend_compiled, passed=true, device_execution=false`。
随后执行设备验证，不能用这个结果替代设备正确性。

## 9. 单卡数值、图重放和完整 CSA L1

正式版对照与第 10 节三方对照二选一。使用 v25 时，先创建下面的运行脚本，
再直接执行第 10 节的任务命令，无需先跑一次正式版任务。

```bash
cat > "$CSA_WORKSPACE/run-csa-repro.sh" <<'BASH'
#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$root/env-csa-repro.sh"
: "${TASK_DEVICE:?请使用 task-submit --device auto}"
case_name=${1:-h131072_b16}
experiment_name=${2:-repro_formal}
label=${3:-$(date +%Y%m%d_%H%M%S)}
shift "$(( $# < 3 ? $# : 3 ))"
control_name=
if [[ $# -gt 0 && "$1" != --device ]]; then
    control_name=$1
    shift
fi
# 队列可能在末尾追加 --device N；这里只核对，设备选择仍以 TASK_DEVICE 为准。
if [[ $# -gt 0 ]]; then
    [[ $# -eq 2 && "$1" == --device && "$2" == "$TASK_DEVICE" ]] || exit 2
fi
base="$CSA_TESTS/csa_mpmd_tnd_20261021"
experiment="$base/experiments/$experiment_name"
extra_args=()
if [[ -n "$control_name" ]]; then
    extra_args+=(--control-mpmd "$base/experiments/$control_name/operator")
fi
out="$CSA_WORKSPACE/results/${experiment_name}_${case_name}_${label}"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
export VLLM_CACHE_ROOT="${out}_vllm_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT"
exec python "$CSA_TESTS/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --workspace-root "$CSA_WORKSPACE" --vllm-root "$CSA_WORKSPACE/vllm" \
  "$experiment/run_case.py" --output "$out" --case "$case_name" \
  --cases "$CSA_WORKSPACE/cases.local.json" --candidate "$experiment/operator" \
  --frozen "$base/sources/84751e34" "${extra_args[@]}"
BASH

printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-csa-repro.sh" h131072_b16 repro_formal first
task-submit --device auto --max-time 900 "$CSA_COMMAND"
```

保留返回的 task-id，用 `task-submit --status <task-id>` 和 `--log <task-id>` 检查实际终态。
队列会分配设备并提供 `TASK_DEVICE`；不要自行固定为 0，也不要在队列外裸跑 NPU。
首次包含 CCEC 编译，几分钟的总作业时间不等于 CSA 算子耗时。

完成后检查 `results/repro_formal_h131072_b16_first/result.json`：

- `stage=complete`、`passed=true`、`device_execution=true`；队列退出码为 0。
- `checks` 中 eager、graph replay、post_l1 三阶段的 output/Top-K/scores/cache/state
  及保护区均通过；整数/索引精确一致，浮点遵守原 `cases.json` 的逐项容差。
- `statistics_us.spmd` 和 `.mpmd` 分别包含 min/mean/max 和全部 5 个样本。
- `windows` 保存对应 `l1_00_spmd/swimlane.json`、`l1_00_mpmd/swimlane.json` 等路径。
  单次泳道看一个窗口即可，不把不同窗口的阶段均值拼成一次执行。

时间取完整 CSA **第一个 incore 开始到最后一个 incore 结束**，包含内部等待和调度空档。
两侧同卡同进程、同输入和初态、warmup 3 次后交替采 5 窗；不累加重叠 task 时长，
不把 Host 编译时间计入 L1。主计时不启用 `--dep-gen`；需要依赖详情时另做诊断。

## 10. 重现已提交的 v25 候选及八档结果

正式入口目前没有采用 v18～v25 的整套改动。若需复现 v25，
按下面顺序从 Git 中的生成器重新建立私有包，而不是寻找未提交的 `experiments/`：

```bash
python "$CSA_CASES/prepare_experiment.py" v17_upstream_fixes
python "$CSA_CASES/prepare_priority_fusion.py" v18_priority_fusion
python "$CSA_CASES/prepare_query_group.py" v19_query_group v18_priority_fusion
python "$CSA_CASES/prepare_query_score_ready.py" v21_query_release_priority v19_query_group --release-query-priority
python "$CSA_CASES/prepare_front_q_priority.py" v22_front_q_priority v21_query_release_priority
python "$CSA_CASES/prepare_query_weights.py" v23_query_weights v22_front_q_priority
python "$CSA_CASES/prepare_score_live_half.py" v24_score_live_half v23_query_weights
python "$CSA_CASES/prepare_score_leaf_rotation.py" v25_score_leaf_rotation v24_score_live_half

printf -v CSA_COMMAND '%q ' bash "$CSA_WORKSPACE/run-csa-repro.sh" h131072_b16 v25_score_leaf_rotation first repro_formal
task-submit --device auto --max-time 1200 "$CSA_COMMAND"
```

每个实验目录只能生成一次；排队和执行期间不得编辑其源码。v26 是本文时点未提交、未采用
的本地候选，不属于本次可公开复现范围。
运行脚本的可选第 4 个参数指定正式 MPMD 控制组；上述命令在同一进程中比较冻结 SPMD、
v25 和 `repro_formal`。控制组的时间位于 `statistics_us.mpmd_control`，数值及保护区
也接受相同的三阶段检查。

八档名称为 `h131072_b1`、`h131072_b4`、`h131072_b8`、`h131072_b16`、
`h131072_b24`、`h131072_b28`、`h8192_b16`、`h8192_b32`。
将第 9 节命令的场景名逐一替换即可；每档有自己的固定混长和 padding 配置。

固定场景及既有单卡结果都在上述 vLLM-Ascend 提交中：

- [公共工具链发布及设备检查](csa_mpmd_tnd_20261021/upstream_refresh_20261021/README.md)
- [固定输入和容差](csa_mpmd_tnd_20261021/cases.json)
- [v25 八档 min/mean/max、全部样本和正确性摘要](csa_mpmd_tnd_20261021/score_leaf_rotation_matrix.json)

## 11. 独立环境的单卡验收结果

已在同一 A3 主机新建独立源码、Python venv 和编译产物目录，从固定源码重新构建
PyPTO、Simpler、LLVM/PTOAS、vLLM-Ascend Native 扩展和上述 14 个 ACLNN 自定义算子。
使用 CANN 9.2.0-beta.2；父/子进程模块路径检查、完整 CPU 后端编译和设备验证均通过。
设备任务 `task_20261021_233209_335615529497` 退出码为 0。

128K/B16 的完整 CSA L1 结果如下，每种实现 warmup 3 次后交替采样 5 次：

| 实现 | min（μs） | mean（μs） | max（μs） |
| --- | --- | --- | --- |
| 冻结 TND 单 SPMD | 713.88 | 719.708 | 727.66 |
| 正式 TND MPMD | 893.16 | 901.096 | 912.54 |
| v25 TND MPMD | 856.30 | 871.728 | 881.92 |

三种实现的 eager、NPUGraph replay、post-L1 共 9 组检查全部通过，每组包含
100 或 101 项保护区检查。正式 MPMD 与 v25 对冻结 SPMD 的 output、Top-K、scores、
SWA cache、compressed cache、compressor state、Indexer key/scale 及 Indexer state
在本场景逐 bit 一致；输入包含第 8 节的混合 S1～S6、变长历史和 9 行 padding。

接收方按第 9 节运行正式版，或按第 10 节运行三方对照，在本机保存 `result.json`
及对应 L1 泳道。要求数值与保护检查通过；绝对时间受硬件频率、驱动和系统负载影响，
不要求复现到上述微秒值。新增独立环境实测覆盖 128K/B16，其余七档的命令可按需执行。

## 附录：当前 Python 环境的实测依赖版本

执行这个代码块后，再回到第 4 节安装。Triton 两个包在第 4 节按顺序单独安装。

```bash
cat > "$CSA_WORKSPACE/requirements-csa-observed.txt" <<'REQ'
agent-detector==2.0.0
aiofiles==25.1.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==1.7.0
anyio==4.15.1
apache-tvm-ffi==0.1.14.post0
arctic-inference==0.1.1
astor==0.8.1
async-timeout==5.0.1
attrs==24.2.0
blake3==1.0.9
blinker==1.9.0
cachetools==7.2.0
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.5.0
cloudpickle==3.1.1
cmake==4.4.2
compressed-tensors==0.17.0
cryptography==50.0.1
decorator==5.1.1
depyf==0.20.0
detect-installer==0.2.1
dill==0.4.1
dnspython==2.8.0
docstring-parser==0.18.0
einops==0.8.2
email-validator==2.3.0
exceptiongroup==1.3.1
execnet==2.1.2
fastapi==0.136.3
fastapi-cli==0.0.32
fastapi-cloud-cli==0.26.0
fastar==0.12.0
filelock==4.0.1
flask==3.1.3
frozenlist==1.8.0
fsspec==2026.9.0
googleapis-common-protos==1.75.3
grpcio==1.84.0
h11==0.16.0
h2==4.4.1
hf-xet==1.6.0
hpack==4.2.0
httpcore==1.0.9
httpcore2==2.13.0
httptools==0.8.0
httpx==0.28.1
httpx2==2.13.0
huggingface-hub==1.32.0
hypercorn==0.18.0
hyperframe==6.1.0
idna==3.20
ijson==3.5.1
iniconfig==2.3.0
interegular==0.3.3
itsdangerous==2.2.0
jinja2==3.1.6
jiter==0.17.0
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==1.7.6
llvmlite==0.49.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
markupsafe==3.0.3
mcp==2.2.0
mcp-types==2.2.0
mdurl==0.1.2
memcache-hybrid==1.2.0
memfabric-hybrid==1.2.0
mistral-common==1.11.7
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgpack==1.2.2
msgspec==0.21.1
multidict==6.9.0
nanobind==2.15.0
networkx==3.4.2
ninja==1.13.0
numba==0.67.0
numpy==2.2.6
openai==3.16.2
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
opentelemetry-api==1.44.0
opentelemetry-exporter-otlp==1.44.0
opentelemetry-exporter-otlp-proto-common==1.44.0
opentelemetry-exporter-otlp-proto-grpc==1.44.0
opentelemetry-exporter-otlp-proto-http==1.44.0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
opentelemetry-semantic-conventions-ai==0.5.1
outlines-core==0.2.14
packaging==26.3
pandas==2.3.3
pandas-stubs==2.3.3.260113
partial-json-parser==0.2.1.1.post7
pathspec==1.1.1
pillow==12.3.0
pip==23.0.1
pluggy==1.6.0
priority==2.0.0
prometheus-client==0.26.0
prometheus-fastapi-instrumentator==8.1.0
propcache==0.5.4
protobuf==7.36.2
psutil==6.0.0
py-cpuinfo==9.0.0
pybase64==1.5.0
pybind11==3.1.0
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.5
pydantic-core==2.46.5
pydantic-extra-types==2.11.1
pydantic-settings==2.15.0
pygments==2.21.0
pyjwt==2.14.0
pytest==8.3.2
pytest-xdist==3.6.1
python-dateutil==2.9.0.post0
python-dotenv==1.2.3
python-json-logger==4.2.0
python-multipart==0.0.32
pytz==2026.3.post1
pyyaml==6.0.3
pyzmq==27.2.0
quart==0.20.0
referencing==0.37.0
regex==2026.9.10
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.5
rignore==0.8.1
rpds-py==0.30.0
safetensors==0.8.0
scikit-build-core==1.0.3
scipy==1.13.1
semantic-version==2.10.0
sentencepiece==0.2.2
sentry-sdk==2.69.2
setproctitle==1.3.7
setuptools==79.0.1
setuptools-rust==1.13.0
setuptools-scm==10.2.3
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sse-starlette==3.4.11
starlette==1.6.0
supervisor==4.3.0
sympy==1.14.0
taskgroup==0.2.2
tiktoken==0.14.0
tokenizers==0.22.2
tomli==2.4.1
torch==2.10.0
torch-npu==2.10.0.post2
torchaudio==2.10.0
torchvision==0.25.0
tqdm==4.70.1
transformers==5.14.1
truststore==0.10.4
typer==0.27.2
types-pytz==2026.3.1.20260727
typing-extensions==4.16.0
typing-inspection==0.4.4
tzdata==2026.4
urllib3==2.8.0
uvicorn==0.53.0
uvloop==0.22.1
vcs-versioning==2.4.1
watchfiles==1.3.0
websockets==16.1.1
werkzeug==3.1.8
wheel==0.48.0
wsproto==1.3.2
xgrammar==0.2.7
yarl==1.25.1
REQ
```
