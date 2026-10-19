# CSA 单 SPMD 配套工具链复现（2026-10-19）

本目录固定已经用于 CSA 单 SPMD 验证的 PyPTO、Simpler 和 PTOAS 源码组合。
三仓分支均为 `dsv4-csa-spmd-repro-20261019`，准确提交号见
[versions.json](versions.json)。分支用于检索，重现实验时以记录的提交号为准。

正式动态算子在
[`deepseek_v4_flash_csa_spmd`](../../../vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/)。
本次发布不改变其服务路由，也不将 CPU 编译成功当作新增真机精度或性能结果。

## 1. 版本和前置条件

| 项目 | 本次对应版本或要求 |
| --- | --- |
| 设备 | Ascend A3；本会话使用 24 AIC、48 AIV |
| CANN | 9.2.0-beta.2；会话最初的 9.0 环境已不是当前实测环境 |
| Python / Torch / torch_npu | 3.10.19 / 2.10.0 / 2.10.0.post2 |
| PyPTO | 保留当前调试分支基础，补齐 CSA 使用的本地编译器修复 |
| Simpler | `57e499238fd4ee3548947ad860be456aaeeb16b5`，256 Tensor + 16 scalar 配套 ABI |
| PTO-ISA | Simpler 管理的 `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`；该源码树无本地修改 |
| PTOAS | 本分支为 0.66 基础的配套修复版；不能只用 `--version` 判断源码是否一致 |
| LLVM/MLIR | `vpto-dev/llvm-project` 的 `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`，LLVM 19.1 |
| CANN cmake | `master-055` 对应 `a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70` |
| Host 编译器 | 本次使用 GCC 15.2 |
| Python 构建依赖 | nanobind 2.15.0、scikit-build-core 1.0.3、cmake 4.4.2、ninja 1.13.0 |
| 模型 | 正式 vllm-ascend W8A8 权重 `DeepSeek-V4-Flash-0731-w8a8`，不使用 cann_recipe 参考权重 |

CPU 编译检查不需要权重和 NPU。完整真机测试另需要已经可运行 Native
DeepSeek V4 Flash 的 vLLM/vllm-ascend 环境，包括匹配版本的 HcPre/HcPost
等 Native 自定义算子。这三份编译器/运行时仓库不包含模型权重、CANN 安装包
或 Native 算子二进制。

本会话配套 vLLM 源码提交为 `752a3a504485790a2e8491cacbb35c137339ad34`。
请先按所在环境的 DSV4 Native 安装流程准备该依赖；下文只处理本次新增的工具链。
使用自己的工作路径，不需要复制原机器用户名和绝对路径。

## 2. 获取源码

```bash
export CSA_WORKSPACE=/your/workspace/csa-repro
mkdir -p "$CSA_WORKSPACE"
cd "$CSA_WORKSPACE"

git clone --branch dsv4-csa-spmd-repro-20261019 \
  https://github.com/nalinaly/simpler.git simpler
git clone --branch dsv4-csa-spmd-repro-20261019 --recurse-submodules \
  https://github.com/nalinaly/pypto.git pypto
git clone --branch dsv4-csa-spmd-repro-20261019 \
  https://github.com/nalinaly/PTOAS.git PTOAS
git clone --branch dsv4-flash-pto-v0.25.1rc1 \
  https://github.com/nalinaly/vllm-ascend.git vllm-ascend
```

按 `versions.json` 将三仓 checkout 到记录的提交，再执行
`git -C "$CSA_WORKSPACE/pypto" submodule update --init --recursive`。
PyPTO 的 `runtime` 子模块已经指向 nalinaly/simpler 并固定到上述扩容提交；
不要对它执行 `submodule update --remote`。单独克隆的 Simpler 和该子模块
必须是同一版本，避免描述符、派发区及参数转储 ABI 混用。

## 3. 构建和环境绑定

先进入已经准备好 Torch、torch_npu 和 vLLM 的 Python 环境，加载对应 CANN
`set_env.sh`，设置可用的 GCC 15 编译器。所有安装在独立 venv 中完成。
示例并行度为 2，可按机器资源调整；PyPTO 仓内应遵循其资源配置文件。

### 3.1 LLVM 与 PTOAS

已有上述 LLVM 构建时直接复用它。需要从源码构建时：

```bash
cd "$CSA_WORKSPACE"
git clone https://github.com/vpto-dev/llvm-project.git llvm-project
git -C llvm-project checkout fa2fd1f75d742fec91ecf40d98c9c7961647ec42
cmake -G Ninja -S llvm-project/llvm -B llvm-build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=AArch64 -DBUILD_SHARED_LIBS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
cmake --build llvm-build --parallel 2

export LLVM_BUILD_DIR="$CSA_WORKSPACE/llvm-build"
cmake -G Ninja -S "$CSA_WORKSPACE/PTOAS" -B "$CSA_WORKSPACE/PTOAS/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DPTO_ENABLE_PYTHON_BINDING=OFF
cmake --build "$CSA_WORKSPACE/PTOAS/build" --target ptoas-native --parallel 2
export PTOAS_ROOT="$CSA_WORKSPACE/PTOAS/build"
export PATH="$PTOAS_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
"$PTOAS_ROOT/bin/ptoas" --version
```

该分支提供复用正式 `runPTOAS` 驱动的 native CLI，生成
`build/bin/ptoas`，不需要 LLVM Python binding。底层编译流水保持一致。
构建时需能获取 CANN cmake；若 GitCode 无法访问，可以预先准备上表版本的
`cann-cmake` 源码目录，并传入
`-DCANN_3RD_LIB_PATH=/your/dependency-parent`，
使其下存在 `cann-cmake/function/prepare.cmake`。不要复制其他工作区的 PTOAS
目标文件或将来源不明的编译产物作为本分支验证结果。

### 3.2 Simpler

在 Simpler 自己的 venv 安装。构建会按照该仓库的配置准备运行时和固定的 PTO-ISA。

```bash
cd "$CSA_WORKSPACE/simpler"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install nanobind==2.15.0 scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install --no-build-isolation -e .
deactivate
```

后续回到准备好 Torch、torch_npu 和 vLLM 的 Python 环境。
`bootstrap.py` 会定位 Simpler 自己的 `.venv/lib/pythonX.Y/site-packages`
中的 `_task_interface`，因此 Simpler 和测试进程必须使用相同的 Python 主/次版本。
已有的 Native CANN 环境变量仍需保留。

### 3.3 PyPTO

```bash
cd "$CSA_WORKSPACE/pypto"
python -m pip install nanobind==2.15.0 scikit-build-core==1.0.3 cmake==4.4.2 ninja==1.13.0
source .claude/skills/testing/load-env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dnanobind_DIR="$(python -m nanobind --cmake_dir)" \
  -DPYPTO_BUILD_TORCH_NPU=ON
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
```

这是原生 CMake 构建，PyPTO core 及 torch_npu adapter 应出现在
`python/pypto/` 下。本目录启动器直接加载它们，无须依赖其他机器的 editable
钩子或复制旧 `.so`。正常源码安装方式仍可使用，但本复现入口要求上述本地构建产物。

## 4. 运行已有 CSA 测试

[bootstrap.py](bootstrap.py) 只负责当前测试进程的包定位。它将历史
`dsv4_csa_env.activate()` 替换为明确的源码绑定，不修改计算、比较门槛和 profiling。
可选 `--vllm-root /path/to/vllm` 用于明确绑定配套 vLLM 源码。
没有此参数时沿用当前 Python 环境中的 vLLM。

准备路径：

```bash
export CSA_REPO="$CSA_WORKSPACE/vllm-ascend"
export CSA_TESTS="$CSA_REPO/tests/pypto_test"
export CSA_FULL="$CSA_TESTS/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
export CSA_PAR="$CSA_FULL/parallel_20261018"
export CSA_SOURCE="$CSA_REPO/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PTO_CSA_VARIANT=performance
```

先做不占卡的完整动态编译：

```bash
python "$CSA_TESTS/csa_toolchain_repro_20261019/bootstrap.py" \
  --pypto-root "$CSA_WORKSPACE/pypto" \
  --simpler-root "$CSA_WORKSPACE/simpler" \
  --ptoas-root "$PTOAS_ROOT" \
  "$CSA_PAR/mgather_physical_plan_retest_20261019/check_cpu.py" \
  --spmd-source "$CSA_SOURCE" --strategy long \
  --controls-json "$CSA_TESTS/csa_toolchain_repro_20261019/cpu_long_controls.json" \
  --output "$CSA_WORKSPACE/results/cpu-long"
```

检查输出 `result.json` 中的通过状态、`single_mix`、动态轴以及实际 import 路径。
切换为 `--strategy medium` 并去掉 `--controls-json` 可检查短历史编译分支。
输出目录需要是新目录。
CPU 编译通过不能证明设备精度、核间同步或性能通过。

真机测试使用原完整测试入口。下面以 128K/B16/S6 为例，使用所在机器的
`task-submit --device auto` 队列提交该 Python 命令；队列内须加载上述环境。
历史 runner 的模型路径仍是 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，
复现前需将正确的正式权重放在该路径。该入口默认 S6。

```bash
python "$CSA_TESTS/csa_toolchain_repro_20261019/bootstrap.py" \
  --pypto-root "$CSA_WORKSPACE/pypto" \
  --simpler-root "$CSA_WORKSPACE/simpler" \
  --ptoas-root "$PTOAS_ROOT" \
  "$CSA_FULL/run_device.py" --spmd-source "$CSA_SOURCE" \
  --output "$CSA_WORKSPACE/results/h131072-b16" \
  --history 131072 --batch 16 --l1-windows 5 --wait-backoff \
  --metadata-replay --padding-replay
```

七个 S6 回归规格是 128K/B4、B8、B16、B24、B28，8K/B16、B32。
计时取完整 CSA 最早 incore 开始到最晚 incore 结束，比较同轮多次采样的中位数。
性能目标和已验证范围仍以原优化日志为准；本次发布不代表七档均达到 30% 降耗时目标。

## 5. 补丁范围与验证记录

- PyPTO：Group InOut 别名、动态 reshape 元数据、inline Tile 注解、signed FloorMod、
  While 内存复用遍历、分支别名/生存期/分配窗口、PTOAS view 代码生成、普通 GM
  notify/wait，以及已提交的常量 slice 和死 Tile 传递闭环处理。
- Simpler：256 Tensor + 16 scalar，参数转储多字位图、派发及图缓存布局同步；
  直接发布此前已验证的提交，没有另外修改调度算法。
- PTOAS：保留 Acc 连续列窗口、INT8 分形 L1 区域及 TEXTRACT 范围分析，
  修复 CFG 并行赋值；GM `MemoryDependentAnalyzer.cpp` 恢复为实测
  `toolchain_v3` 使用的版本，排除未采用的 GM 列分片试验。
- 当前 PyPTO 的 `runtime` URL 和提交号一起固定；PTO-ISA 是干净的官方 pin。
- 上游问题仍通过 issue 跟踪，本次不创建上游修改 PR。

具体检查与历史设备验证边界记录在 [verification.json](verification.json)。
仓内仅保存源码、文档和轻量记录；权重、二进制、完整编译缓存及大快照保留在原机器。
