> **长期保留的过程日志；更新至 2026-09-26。**
> 当前合同与待办以 [执行清单](DSV4_FLASH_CSA_TASK_CHECKLIST.md) 为准，
> 可运行入口见 [验证 README](README.md)，本轮过程从 [第 99 节](#log-20260926) 开始。
> 第 1～98 节及其旧版开头完整保留，记录的是各自日期、源码、工具链和输入下的历史；
> 其中“当前”、PASS、待办和修改权限均不自动沿用到本轮，尤其不能将旧 eager 结果用于图模式验收。
> 历史链接对应的过时脚本或产物可能已清理，可按所属提交从 Git 查看；保留本日志不恢复这些旧入口。
> 后续每个阶段在本文件追加修改原因、提交、测试配置与结果、结论边界和下一步。

<!-- 以下保留删除前的完整历史原文；2026-09-26 的新增记录在文末。 -->

> **2026-09-23 基线已迁至官方 v0.25.1rc1。** 当前入口为[基线迁移说明](BASELINE_MIGRATION_V0251RC1.md)。
> 下方历史 PASS 对应旧基线；本次 release 尚未进行真机数值验收。历史脚本需按新接口适配。

# DeepSeek-V4 Flash CSA 本机验证过程记录

- 开始日期：2026-09-21；时间按北京时间记录。
- 最后更新：2026-09-22，最新参考18组、B4/B8连续轨迹与B4/B40跨流延迟通过；其余轨迹定位输出边界。
- 工作目录：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto`。
- 计划：[DSV4_FLASH_CSA_VALIDATION_PLAN.md](DSV4_FLASH_CSA_VALIDATION_PLAN.md)。
- 本文件记录已经执行的操作、失败与修正；计划中的用例不因列在这里而视为通过。
- 验证边界：CSA `attention.forward`；全流程 ND；不包含外层 HC pre/post、外层 RMS、MoE。
- 当前机器有16个Ascend910_9392逻辑设备；硬件任务均通过`task-submit`分配设备。
- 当前继续P0～P4单/双卡开发验证，未启动完整模型或16rank；正式目标权重等待用户通知下载完成。

## 1. 当前状态

| 项目 | 状态 | 实际完成范围 |
| --- | --- | --- |
| P0 环境与参考权重核对 | 部分通过 | CANN9.0、Torch/torch_npu2.10及debug PyPTO/Simpler已恢复；参考ND通过，正式ModelSlim目标权重待验 |
| P1 metadata transport 预检查 | PASS | 两张卡分别通过合成 metadata 的 eager、最小 NPU Graph 和 cache 精确对照 |
| P1 shared storage 预检查 | PASS | INT8 K/FP16 scale 共享 allocation，非零 offset，页 padding，图 A→B→A，保护区精确对照 |
| P1 原生 M01～M12 | PASS（metadata范围） | 六档BS、五组native builder/ForwardContext、图A→B→A、stride/offset、slot、负例均通过；证据见第13节 |
| P2 完整单层 Native/PTO 对照 | 参考18/18 PASS；正式目标待验 | reference_matrix_v5共324项通过，包含Indexer RMS及八行池化修正，见第60节 |
| P3 连续接受轨迹与状态 | PARTIAL | 当前版本B4/B8 mixed100及B4三种接受边界各eager100+graph100通过；B16/24/32 step63、B40 step46仅输出超差；B4/B40三类Native生产stage延迟各通过，见第60～62节 |
| P4 两个真实 DP rank | 部分通过（metadata） | 真实TP1/DP2的同步、dispatcher与非空rank五组probe已过；完整CSA、PD分支和真实dummy路径未验 |
| P5 16 卡整模型 | NOT_RUN | 当前硬件具备16逻辑设备；前置验证和目标权重完成后自行先启动P再启动D |

本机P0～P4尚未全部完成。PASS只覆盖表中声明范围；参考权重结果不替代目标checkpoint或整模型验收。
当前环境见[CANN9.0环境记录](handoff/ENVIRONMENT_CANN90.md)。以下第2～16节保留旧机器历史，
其中CANN9.2、两卡、旧绝对路径与迁移决定不是当前运行配置；第17节起记录当前机器。

## 2. 旧机器固定环境与源码（历史）

| 项目 | 实际值/来源 |
| --- | --- |
| PTO 工具链与来源 | `source /mnt/workspace/inductor/pto_eager/env.sh`，随后由辅助脚本选择对应 PyPTO/Simpler |
| 后续模型验证解释器 | `/mnt/workspace/inductor/vllm-ascend/.venv/bin/python`，Python 3.11.4，复用原有模型依赖 |
| 早期预检查解释器 | `/mnt/workspace/inductor/pto_eager/.venv/bin/python`；第5、6节已记录的小核结果使用此解释器 |
| PyTorch / torch_npu | `2.12.0+cpu` / `2.12.0+git5462a1b`；NPU 由 torch_npu 提供 |
| CANN | `/home/developer/Ascend/cann-9.2.0` |
| PTOAS | `install-v0.57-llvm21-cann9.2-clean`，由 pto_eager 环境选择 |
| PyPTO / Simpler | `pto_eager/pypto` / `pto_eager/simpler`；实际 import 路径由环境辅助脚本检查 |
| PyPTO runtime | `tensormap_and_ringbuffer`，平台 `a2a3` |
| 硬件 | 两张 `Ascend910_9362`，逻辑 device 0、1，每张约 64 GB HBM |
| vLLM-Ascend | 官方 `34bb51f93724c565362f5108f5226303e1b56cad`，分支 `dsv4-flash-pto` |
| vLLM | 验证基线 `84030bbe3d74d99bad477a3d2e37a973ccd8865c` |
| vLLM 本地 checkout | `.cache/csa/vllm-84030bbe`，独立 worktree |
| pypto-lib | `205255b4770ee84dfa176bcbc7bbef651953c7e1` |
| 格式目标 | Native/PTO 均 ND；`additional_config.weight_nz_mode=0` |

完整环境初始快照见 [manifest.json](results/local_20260921/manifest.json)。快照中的 import 失败项是生成时的结果，
后续变化以本记录及新增结果文件为准；不能把一次 import 成功解释为整个运行环境已验收。

`pto_eager/pypto` 原本已有 `_kernel_abi.py`、`torch/shutdown.py` 的本地修改。本次保留原样，
未用重置仓库的方式处理环境问题。`pypto-lib` 未修改；未复制原工作区的私有 CSA adapter。

### 2.1 Python 环境准备过程

1. 最初将用户要求的 PTO 环境理解成必须使用 pto_eager 的解释器，后来发现这一限制不必要。
   正确复用方式见第2.3节；早期建立的任务 `.venv` 不用于后续验证。
2. 处理用户 site 中其他 PyPTO/Simpler editable hook 抢占路径的问题：
   [dsv4_csa_env.py](dsv4_csa_env.py) 只在当前验证进程内选择 pto_eager 对应的 hook，并检查 import 来源。
3. 将固定提交的 vLLM 以 `VLLM_TARGET_DEVICE=empty` 安装为 editable 包，Ascend Python 包以
   `COMPILE_CUSTOM_KERNELS=0` 安装；该步骤只准备 Python 包和插件入口，不代表原生算子已可运行。
4. 缺少的依赖装入 pto_eager `.venv`。没有重装 PyTorch/torch_npu。
5. aarch64 上 sklearn 的 libgomp 曾报 `cannot allocate memory in static TLS block`；
   原生 import 检查的启动命令中预加载对应 libgomp，未修改业务代码绕过 import。
6. 最初从所查 pip index 获取了 triton-ascend 3.2.0，随后在官方 GitHub Release 找到3.2.2 wheel。
   19:42 前后已将 pto_eager `.venv` 中的3.2.0升级到3.2.2。该安装已完成，不以“准备安装”记录。
   原有模型环境此前就装有3.2.2，本轮查明后未修改其依赖。
7. 依赖仍存在未验收的组合，例如原生 Ascend requirements 与所固定 vLLM 的 FastAPI 版本范围不同；
   本机单层验证不据此宣称 HTTP 服务环境通过。

安装日志保留在本地 `.cache/csa/`，包括 `install-vllm.log`、`install-ascend-python.log`、
`pip-pto-eager.stdout.log`、`pip-triton.stdout.log`、`pip-vllm-extra*.stdout.log`、`install-extra*.log`。

### 2.2 原生扩展实验构建

- 官方 `CMakeLists.txt` 要求 Torch 2.10.0；pto_eager 环境实际为 2.12.0。
- 在 `.cache/csa/native-source` 中创建构建入口，保留原始 C++ 源码；仅将副本中的版本检查从 fatal 改为显式 warning。
- 原仓库的 `CMakeLists.txt` 没有修改。该构建属于本机兼容性实验，不代表官方声明支持 Torch 2.12。
- `SOC_VERSION=ascend910_9362`，`cmake --build ... -j 8` 已成功。
- 产物安装在 `.cache/csa/native-install`，实际 `.so` 已通过 Python 扩展加载。
- 后续直接导入 `dsa_v1` 曾遇到 `DeviceOperator` 与 `ops` 的循环 import；先初始化 `vllm_ascend.ops` 后推进到 Triton driver 初始化。
- 19:38 的历史失败：triton-ascend 3.2.0 编译 `npu_utils.cpp` 时引用的
  `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 不存在于当前 CANN headers。
  这是当时所选 Python 依赖环境的兼容问题，不能计为 metadata 或 CSA 数值失败。
- 19:46 复用原有模型环境后，原生 `dsa_v1` import 已通过；见第2.3节。
- 19:58 后已执行原生 metadata smoke：SAS/QLI 在加载现有 vendor 包后通过；CompressorMetadata 缺失。
  后续处理见第10节；扩展可加载不等于全部 CANN vendor 算子版本匹配。

构建证据：

- `.cache/csa/native-build-compatibility.patch`：唯一构建入口差异。
- `.cache/csa/native-configure-command.json`：实际 CMake 参数。
- `.cache/csa/native-configure.log`、`native-build.log`、`native-install.log`：完整输出。
- `.cache/csa/native-import.log`：原生初始化的实际失败堆栈。
- [environment_packages.json](results/local_20260921/environment_packages.json)：本次准备后的关键包版本。
- [native_build.json](results/local_20260921/native_build.json)：产物 hash、构建差异及未完成项。

### 2.3 用户提醒后复核 Qwen 运行环境

用户指出“之前的 Qwen 不是跑过 vLLM-Ascend 么”，据此暂停继续安装依赖，检查现有运行证据。

已确认：

- `profiling/qwen3_14b_native_vs_pypto_full_decode_aclgraph_3steps_20260920/comparison.md`
  记录 Native/PTO 均生成 `[12095, 13, 3555, 374]`，各有3次完整 decode ACL Graph execute。
- 两侧 `profiler_info_0.json` 都记录 torch_npu `2.12.0+git5462a1b`、CANN `9.2.0`。
- `vllm-ascend/.venv` 已有 vLLM editable 安装、triton-ascend3.2.2、Transformers5.14.1、FastAPI0.136.3等模型依赖。
- Qwen 的 `activate_pto_eager()` 负责选择 PTO editable hook，没有要求模型解释器必须为 pto_eager `.venv`。
- 19:45 实测使用原有模型解释器、显式源码路径和 PTO helper，成功导入 vLLM、Qwen分支、PyPTO、Simpler。
- 19:46 将源码路径切换为 DSV4分支与固定的 vLLM `84030bbe`，成功导入本次构建的原生扩展、`ops`、`dsa_v1`。
  日志：`.cache/csa/native-import-existing-env.log`。

结论：早期仅根据一个 Python 进程找不到 vLLM 就开始补装依赖，漏查了现有模型环境。
后续复用原有依赖环境，同时显式固定 DSV4 源码路径；不会因为复用解释器而导入旧私有 Ascend 源码。
[dsv4_csa_env.py](dsv4_csa_env.py) 已去掉必须使用 PTO 自有解释器的限制，保留源码路径检查。

前述补装发生在 pto_eager `.venv`，需如实保留记录；本轮环境复核未修改原有 `vllm-ascend/.venv`。
已安装包没有在未检查影响的情况下批量卸载。原生 metadata **导入通过**尚不等于实际 builder/算子执行通过。

官方3.2.2包来源为
[Triton-Ascend v3.2.2 Release](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.2)，
对应历史安装日志 `.cache/csa/install-triton322.log`。

## 3. 本机量化 checkpoint 核对

用户提供 `/usr/.devenv` 后，只读检查得到：

```text
/usr/.devenv/models/DeepSeek-V4-Flash-W8A8
```

该目录存在 config、权重索引及 46 个 safetensors shard 文件。没有读取或复制完整模型，
也没有把它认定为最终 16 卡目标 checkpoint。

- Layer 2 是第一个 `compress_ratio=4` 的 CSA 层。
- 找到该层 `layers.2.attn.*` 的 21 个 tensor，集中在 `model-00004-of-00046.safetensors`。
- 选中 tensor 的 payload 总计 176,890,368 bytes，约 168.7 MiB。
- 检查了 header 中 shape、dtype、offset 与文件长度；尚未加载 payload 做数值/完整性校验。
- 本地配置是 compressed-tensors W8A8：动态 per-token activation、per-channel weight scale，部分投影保留 BF16。
- 该量化描述与目标脚本的 `quantization=ascend` 不能直接视为相同；后续需显式映射并核对每个模块。
- ND 只约束内存格式，不意味着将 W8A8 改成全 BF16。

证据：[checkpoint_inventory.json](results/local_20260921/checkpoint_inventory.json)。

## 4. 非连续 Tensor / 共享 storage GAP

### 4.1 源码确认的约束

1. PyPTO Torch interop 当前要求 contiguous strided Tensor，且格式为 ND/NCHW；不自动复制，也不自动把任意 PyTorch stride 传入 kernel。
2. 原生 runner 的 `_adjust_kv_layout` 用 `as_strided` 创建 cache view，第一维 stride 来自真实 page bytes，
   不能通过逻辑 shape 相乘替代。
3. Indexer K 与 scale 可位于同一 allocation 的不同页内区域；A3 的 K 是 INT8，scale 是 FP16。
4. pypto-lib 参考入口的 indexer scale 参数是 FP32；需要明确转换语义，不能把 FP16 字节直接当 FP32 读取。
5. PyPTO interop 会拒绝部分重叠且可写的参数；将两个 view 各自展平为覆盖整个 storage 的可写 Tensor 并不能解决共享存储问题。

源码位置：

- `pto_eager/pypto/python/pypto/torch/interop.py::_validate_tensor/_describe_tensor/_validate_aliases`。
- `vllm_ascend/worker/model_runner_v1.py::_adjust_kv_layout/_reshape_kv_cache_tensors`。
- `vllm_ascend/models/deepseek_v4/indexer.py`、`vllm_ascend/device/device_op.py`。
- `pypto-lib/models/deepseek_v4_flash_dspark/decode_csa.py::_decode_csa_tp1`。

### 4.2 当前采用的验证方向

- 静态只读权重可以在初始化时准备连续 ND 布局。
- metadata 内容继续留在 device；host 只读取 shape、stride、storage offset、容量等 Tensor 描述信息。
- 可变 cache 保留原 allocation，由 adapter 提供物理布局参数，kernel 按真实 stride/offset 寻址。
- 共享 allocation 作为一个 InOut 参数传入；核内按 K、scale 子区域的 dtype 读取并写回。
- 不对整个 KV cache 每步执行 `.contiguous()`；这种复制还需要解决原位写回、alias、graph 地址和额外带宽问题。
- 新方案目前仅通过下述小核验证；完整 CSA 的各子 kernel 尚待适配。

## 5. Metadata transport 探针

代码：[dsv4_csa_metadata_kernel.py](dsv4_csa_metadata_kernel.py)、
[dsv4_csa_transport_smoke.py](dsv4_csa_transport_smoke.py)。

### 5.1 已执行输入与检查

- BS=4，每请求 query=6，实际 T=24，buffer capacity=32。
- 初始位置覆盖 131071、131072、131073、131078；device 输入包含请求边界、长度、位置、页表和二元 slot。
- table row stride=4112，cache page stride=40（该探针为 INT32 sentinel cache，单位是 element）。
- PTO 从 device 读取上述内容，输出 16 列诊断信息，并按 slot 修改 sentinel cache。
- eager 与最小 NPU Graph replay 均与独立 CPU 预期逐值比较；也比较 padding/未写区域。

### 5.2 发现的错误与修正

首版由多个 token worker 并发写 4-byte sentinel，相邻写入共享同一个缓存行，出现 cache 值不匹配。
输出诊断行正确仍不足以证明 cache 更新正确。

修正：读任务完成后，由一个诊断 commit task 写入这些小标量，并显式设置 `deps=[read_tid]`。
该串行 commit 只用于探针。正式 KV 写回应按完整对齐行/页划分任务，不能直接沿用探针写法作为性能实现。

原始失败保留在
[transport_scalar_cache_write_race.json](results/local_20260921/failures/transport_scalar_cache_write_race.json)。

### 5.3 实测结果与边界

| Device | Eager | Graph replay | Cache/padding | 证据 |
| --- | --- | --- | --- | --- |
| 0 | 精确通过 | 精确通过 | 精确通过 | [transport_device0.json](results/local_20260921/transport_device0.json) |
| 1 | 精确通过 | 精确通过 | 精确通过 | [transport_device1.json](results/local_20260921/transport_device1.json) |

这些是合成 metadata 预检查。没有覆盖原生 builder、五个 cache group、全部 BS 或两个 rank 的 DP 同步。

## 6. Indexer K/scale 共享存储探针

代码：[dsv4_csa_shared_storage_kernel.py](dsv4_csa_shared_storage_kernel.py)、
[dsv4_csa_shared_storage_smoke.py](dsv4_csa_shared_storage_smoke.py)。

### 6.1 已执行输入与检查

- 7 个 physical page；每页 32 个 token，每个 K 向量 128 个 INT8 element。
- 页内 K 占 4096 bytes；随后是 32 个 FP16 scale，占 64 bytes。
- 测试 page bytes=4160，以及额外填充到 32768 的布局。
- 整个输入 view 的 storage offset 为 128 bytes，allocation 前后均有保护区。
- CPU 端使用非连续、共享 storage 的 K/scale view 生成独立预期；PTO 入口只传一个连续原始存储 view。
- PTO 读取 K/scale 后计算诊断值，再原位更新 K 和 scale；逐字节比较完整 allocation，包含页 padding 和前后保护区。
- 同一张图、相同 device 地址，依次填入 A、B、A 三组内容并 replay；每次均比较输出和写回。

实现时显式使用 A3 支持的 `INT8 → FP16 → FP32` 转换链；INT8 的全部取值均可由 FP16 精确表示。
这是数值转换；scale 区域的 `reinterpret_view` 则只解释同一组 FP16 字节，两者不能混淆。

### 6.2 实测结果

| Device | Page bytes | Eager | Graph A→B→A | 保护区/页 padding | 证据 |
| --- | --- | --- | --- | --- | --- |
| 0 | 4160 | 精确通过 | 精确通过 | 精确通过 | [初版结果](results/local_20260921/shared_storage_device0.json) |
| 0 | 32768 | 精确通过 | 精确通过 | 精确通过 | [padded 结果](results/local_20260921/shared_storage_device0_page32768.json) |
| 1 | 4160 | 精确通过 | 精确通过 | 精确通过 | [device 1 结果](results/local_20260921/shared_storage_device1_page4160.json) |

4160/32768 是显式构造的测试布局，不是本次启动完整 runner 后实测得到的最终 cache spec。
结论仅为：单 allocation 参数加核内类型/offset 访问的方向已在 A3 上得到精确验证。
尚不能推出完整 CSA 的 state、TopK、数值或 graph 兼容性已通过。

## 7. 当前复现入口

以下是第5、6节历史小核的复现命令，使用其原始解释器。后续模型验证改用第2.3节说明的组合。

```bash
cd /mnt/workspace/inductor/vllm-ascend-dsv4-flash-pto
source /mnt/workspace/inductor/pto_eager/env.sh

python tests/pypto_test/dsv4_csa_transport_smoke.py \
  --device 0 --output-dir tests/pypto_test/results/local_20260921

python tests/pypto_test/dsv4_csa_transport_smoke.py \
  --device 1 --output-dir tests/pypto_test/results/local_20260921

python tests/pypto_test/dsv4_csa_shared_storage_smoke.py \
  --device 0 --page-bytes 32768 \
  --output-dir tests/pypto_test/results/local_20260921

python tests/pypto_test/dsv4_csa_shared_storage_smoke.py \
  --device 1 --page-bytes 4160 \
  --output-dir tests/pypto_test/results/local_20260921
```

后续正式回归应使用新的结果目录，保留历史失败与源版本。NPU Graph 捕获前先执行 eager，完成 JIT/warmup。
小核脚本里的 CPU 拷贝用于验证结果；它们不属于将来 attention.forward 的热路径。

后续模型验证启动时，先加载 PTO 工具链，再显式调用原有模型环境的解释器：

```bash
source /mnt/workspace/inductor/pto_eager/env.sh
/mnt/workspace/inductor/vllm-ascend/.venv/bin/python <验证脚本及参数>
```

验证脚本必须在导入模型模块前调用环境辅助函数，检查实际源码来源并记录解释器；
上述示意命令不代表完整 CSA 程序已经实现。

## 8. 下一步与更新规则

1. 完成原生 Python 初始化与 CANN 算子匹配检查，记录原生 metadata 实际运行结果。
2. 将原生 BlockTable、metadata builder、ForwardContext 接到 PTO probe，补齐 M01～M12。
3. 按实际 cache spec 获取 stride/offset/alias，替换当前手工构造的布局来源；检查所有 cache group。
4. 加载选中的单层参考权重，保留 W8A8/ND；实现仅 attention.forward 的适配，执行 P2。
5. 继续 P3 连续接受轨迹、原生异步更新链与 P4 两个真实 DP rank。
6. 汇总本机结果和 16 卡待验步骤；P5 与本机结果分开验收。

每次有实质进展，追加：输入/环境变化、命令、结果、证据路径、失败原因及修正、尚未覆盖的范围。
修正后通过必须保留有价值的原失败；不能把不同版本、不同量化或不同布局的结果合并为同一组 PASS。

## 9. 记录与代码检查

- 本记录和计划中的本地 Markdown 链接已检查，均指向现存文件。
- 新增验证脚本已通过 Ruff 检查与格式化检查；本记录与计划已通过 Markdown lint。
- Markdown hook 在本仓库配置为 manual stage，使用 `pre-commit run --hook-stage manual markdownlint --files ...`。

## 10. 19:57～20:10 原生 metadata 实际执行及文档审阅

### 10.1 原生算子 smoke

新增 [dsv4_csa_native_ops_smoke.py](dsv4_csa_native_ops_smoke.py)，使用原有模型解释器，
BS4、query6，历史位置覆盖131071/131072/131073/131078；只检查实际执行，不计为完整P1数值验收。

1. 仅加载PTO环境时，SAS报 `aclnnSparseAttnSharedkvMetadata ... not in libopapi.so`。
   检查发现 `ASCEND_CUSTOM_OPP_PATH` 未设置；已安装 `custom_transformer` 包中存在该接口。
2. 在测试shell中额外source
   `/home/developer/Ascend/cann-9.2.0/opp/vendors/custom_transformer/bin/set_env.bash` 后，
   SAS与QLI v2 metadata均已在NPU执行成功。没有改系统环境文件或重新安装Python依赖。
3. CompressorMetadata仍缺少aclnn接口，因此仅对官方当前源码的 `compressor_metadata` 启动局部构建。
   命令：`bash build.sh --pkg --ops=compressor_metadata --soc=ascend910_93 --vendor_name=csa_validation -j8`。
4. 首次编译遇到CANN9.2新旧 `graph/error_codes.h` include guard相互屏蔽，`ge::graphStatus`未定义。
   使用CMake构建参数预包含CANN9.2原生 `graph/error_codes.h` 重试，未修改CANN安装目录或算子源文件。
   截至本节更新时间，kernel binary已生成，整体构建仍在进行；尚未安装或执行新CompressorMetadata。

结果：

- [初次SAS失败](results/local_20260921/native_sas_metadata_device0.json)。
- [加载vendor后的SAS](results/local_20260921/native_vendor_loaded/native_sas_metadata_device0.json)。
- [QLI v2](results/local_20260921/native_vendor_loaded/native_qli_metadata_device0.json)。
- [CompressorMetadata缺失](results/local_20260921/native_vendor_loaded/native_compressor_metadata_device0.json)。
- 构建日志：`.cache/csa/compressor-metadata-build.log`、`compressor-metadata-build-compat.log`。

### 10.2 原生 builder fixture 的进展与失败

新增真实VllmConfig、DSpark配置和TP1/HCCL初始化入口，未启动整模型或加载drafter权重。

- vLLM模型检查会启动新Python子进程；最初只改父进程 `sys.path`，子进程仍导入旧vLLM。
  helper现将固定源码路径同时写入当前进程的 `PYTHONPATH`，供子进程继承。
- 固定后的ModelConfig解析成功，保留本机checkpoint的 `compressed-tensors` 描述；
  不把这项当作目标 `quantization=ascend` 权重加载验收。
- 完整VllmConfig与真实TP1分布式组初始化已通过。
- 首次builder smoke在原生RoPE初始化时报CPU/NPU输入混用，尚未运行到BlockTable或builder结果校验。
  该fixture还需要与原生模型加载时一样设置初始化device context。
  结果：[native_builder_device0.json](results/local_20260921/native_vendor_loaded/native_builder_device0.json)。

### 10.3 同事的CSA探索文档审阅

按用户要求阅读 [csa-lib-ask.html](csa-lib-ask.html)，保留原文件，
逐项对照当前源码；意见见 [CSA_LIB_ASK_REVIEW.md](CSA_LIB_ASK_REVIEW.md)。

认可非连续cache、共享K/scale、dtype及动态轴检查问题；不接受以下未经证明的推断：

- 转换层存在不可消除的1 ms性能下界。
- 图重放要求所有metadata在图内生成。
- 128行/16640字节是所有vLLM配置的固定页布局。
- 连续无崩溃与trace中有PTO任务就证明数值/状态正确。

同事报告对应的原始CSV、分析JSON/脚本未随本机HTML一起提供，未将其性能数字计作本机实测。
该审阅记录已通过Markdown lint和本地链接检查，不改变P0～P4验收标准。

## 11. 20:15 后：原生 SWA 传参和 CompressorMetadata 进展

用户明确同事文档仅供参考；后续继续原定 P0～P4，不扩展文档审阅范围。

- 原生 builder 的 RoPE device context 修正后，BS4/验6 的真实 BlockTable、slot、seq_lens
  和 start_pos 均通过独立公式检查，见
  [builder v2](results/local_20260921/native_fixture_v2/native_builder_device0.json)。
- [dsv4_csa_native_transport.py](dsv4_csa_native_transport.py) 已完成六档 BS=4/8/16/24/32/40：
  原生 BlockTable → AscendDSAMetadataBuilder → ForwardContext → PTO custom op → NPU。
  eager、同地址 A→B→A 图重放、整份 sentinel cache 和无效 slot/padding 写保护均逐元素一致。
  Host CPU 长度上界特意比 GPU 实际长度多5；PTO 读到的是 GPU 内容。
  结果见 [SWA 六档汇总](results/local_20260921/native_transport_v1/native_swa_transport_device0.json)。
  这只覆盖 SWA group 的 metadata/sentinel，不是完整 attention，也不是完整 P1。
- 官方 `compressor_metadata` 局部构建、打包、安装、NPU 实际调用均成功。
  安装限定在 `.cache/csa/compressor-metadata-install/vendors/csa_validation_transformer`，
  未覆盖系统 vendor。只通过 CMake `-include` 参数处理新旧 graph 头文件冲突，算子源码未改。
  结果见 [本地 CompressorMetadata](results/local_20260921/native_local_vendor/native_compressor_metadata_device0.json)。
  BS4/验6 返回10行容量，其中6行有效，尾部页号为 `-1`；需按压缩输出行的语义处理。

后续原生测试需在已有工具链环境之后，额外加载两个 vendor：

```bash
source /home/developer/Ascend/cann-9.2.0/opp/vendors/custom_transformer/bin/set_env.bash
source .cache/csa/compressor-metadata-install/vendors/csa_validation_transformer/bin/set_env.bash
```

新增 vendor 构建记录：`.cache/csa/compressor-metadata-build-compat.log`、
`.cache/csa/compressor-metadata-package.log`、`.cache/csa/native-compressor-local-execution.log`。
执行器仍使用原有模型环境 `/mnt/workspace/inductor/vllm-ascend/.venv/bin/python`。

## 12. 20:24 后：真实单层权重、原生 cache 布局与新配置差异

新增 [dsv4_csa_native_layout.py](dsv4_csa_native_layout.py)，通过 safetensors 只读取
第2层 CSA 的21个实际 Tensor，共168.7 MiB，没有加载整模型。
使用当前原生 `DeepseekV4Attention`、Ascend Linear、compressed-tensors 量化方法及其加载后处理。
参数加载后先逐元素比对 checkpoint（compressor norm 按原生要求转FP32），再执行原生布局处理。
不把本地 compressed-tensors 权重当作目标16卡 ModelSlim/ascend 权重验收。

独立入口补齐了两项必要初始化，并保留失败记录：

- Ascend 动态量化模块另外分配 weight_offset；checkpoint 没存这些值。
  仅在确认配置为 symmetric 后派生零 offset，其他缺失参数仍报错。
- 与原生 worker 一样调用 `register_ascend_customop(config)`，使用真正的 Ascend Linear。
  缺少这一步时，`wo_a` 的加载后处理找不到分组属性。

[layout v4](results/local_20260921/native_layout_v4/native_layer_layout_device0.json) 已通过真实权重加载、
全参数 NPU format=2（ND）及五组 cache owner/spec/原生 runner `_adjust_kv_layout` 的执行检查。
实际发现其 **indexer 为 BF16 K + FP16 scale，页8256字节**：当时保留了 `indexer_kv_dtype=auto`。
这不是目标INT8布局的PASS。当前 `DeepseekV4Indexer` 将 `auto` 按模型activation dtype解析；
checkpoint里的 `li_cache_scheme=int8` 没有自动覆盖它。

当前A3原生 `DeviceOperator.indexer_quant_scatter` 明确量化到INT8，scale转FP16。
因此后续 Native/PTO 对照共同显式设置 `AttentionConfig(indexer_kv_dtype="int8")`，
并与真实CLI一样执行 `adapt_patch(is_global_patch=True)` 后再构建配置，使用已有INT8类型注册。
计划3.2已补充此差异，16卡启动需一并核对。v4结果保留用于说明默认值问题，不用于INT8数值结论。

五组 probe 新入口为 [dsv4_csa_all_groups.py](dsv4_csa_all_groups.py)：
真实 metadata builder、CompressorMetadata、DeviceMetadataExecutor 外部事件及 graph consumer，
带原生页内stride、非零storage offset、共享K/scale和整份cache guard比对。
压缩slot的展开由PTO小核在device上完成；正在执行首轮，不预先记PASS。

## 13. 五组原生 metadata 矩阵结果

以下结果使用原有模型解释器、原始本地C++扩展、当前源码单独构建的CompressorMetadata，
明确设置INT8 indexer cache。P1没有加载权重内容，也没有调用attention数学计算。

- [BS4及8个拒绝用例](results/local_20260921/native_groups_v5/native_groups_B4_device1.json)。
- [BS8/16/24/32/40汇总](results/local_20260921/native_groups_v4/native_groups_device1.json)。
- [真实参考权重与INT8 cache布局](results/local_20260921/native_layout_v5/native_layer_layout_device0.json)。

| 用例 | 已执行检查 | 结果 |
| --- | --- | --- |
| M01～M03 | 六档BS、query6、异长、历史131071/131072/131073及后续变动 | 离散输出逐元素一致 |
| M04 | 五组不同非连续页号，按各自prefix从ForwardContext取metadata | 一致 |
| M05 | 原生BlockTable swap/move/clear/add，五组sentinel按物理页取值 | 一致；完整CSA状态连续性另属P3 |
| M06 | 实际token以外的图容量、真实token中的无效slot | 未误写，所有padding输出为-1 |
| M07 | 原生runner布局，非零offset，state页padding，共享indexer K/scale | 整份allocation（含保护区）一致 |
| M08 | A3 INT32页号/offset、设备端压缩slot展开、INT64 flat计算 | 有效值一致，无效值保持-1 |
| M09 | CPU乐观长度比GPU实际值多5 | PTO读取GPU实际值 |
| M10 | 固定地址A→B→A；原生DeviceMetadataExecutor跨流ExternalEvent | eager与graph都一致，producer/consumer间无全局sync |
| M11 | ragged query、mixed prefill | 在发射PTO前拒绝 |
| M12 | 错group、逻辑页越界、物理页越界；另测动态轴/位置dtype/token容量 | 在发射PTO前拒绝 |

M12页号检查使用已有的原生CPU allocator mirror，属于测试preflight，未从NPU下载页表，
不主张在生产forward中逐步扫描完整CPU页表。它不等于能检测任意设备内存损坏。
真正热路径里的shape/stride/dtype检查与设备Tensor内容读取分开。

INT8配置的原生cache实测：SWA/main KV每页32768字节；main state也是32768字节，
其中实际两行共16384字节；indexer K/FP16 scale共4160字节；indexer state实际4096字节、页步长4160。
测试统一在前后加128字节保护区，所有Tensor保留实际storage offset。

## 14. 连续轨迹及原生完整forward预执行

- 100步metadata预检查使用原生 `update_num_computed_tokens_for_batch_change` 和
  `correct_optimistic_seq_lens_cpu`，注入 `[1,6,2,5,5]`，第25/75步加入prev_positions换位。
  metadata由修正后的GPU长度生成，随后通过同一张图读取；CPU公式只用于事后断言。
- 首次连续轨迹触发了preflight的逻辑页容量保护：原来的单步页表容量不足以容纳100步继续增长。
  结果保留在 `native_trace_v1`；fixture现按100步最多推进600 token预留页表容量，随后重跑 `native_trace_v2`。
  此轨迹仍是metadata/sentinel预检查，不能替代完整attention cache/state的连续数值比较。
- 新增 [dsv4_csa_native_forward.py](dsv4_csa_native_forward.py)，BS4、query6、完整131072历史，
  真实单层权重，实际分配并初始化所有32768条/请求压缩候选，没有缩小历史。
- 原生forward首次在内部Q norm的 `aclnnRmsNormDynamicQuant` 失败：仓库封装调用双scale/双输出的
  custom ABI；CANN9.2内置同名接口为单scale/单输出，报 `yOut != nullptr`。
  见 [原始失败](results/local_20260921/native_forward_v1/native_forward_B4_L131072_device0.json)。
- 尝试构建仓库同名custom op时，曾出现5输入与4输入定义的冲突；日志
  `.cache/csa/csa-extra-ops-build.log`。21:00前后进一步查明：增量构建只更新了INI，
  `autogen`及`build/custom`下的ops-info JSON仍只有CompressorMetadata，导致编译器仍读取内置RMS定义。
  因此不能将首次构建失败归因于“CANN无法构建该custom op”。现使用仓库生成脚本刷新JSON，
  确认包含本轮全部五个算子后重建，日志 `.cache/csa/csa-full-ops-build.log`。
- 隔离兼容构建脚本 [build_native_cann92.py](build_native_cann92.py) 另行试验按本机CANN9.2头文件
  修正RMS及rotary调用ABI，复用其他原生对象，输出到 `.cache/csa/native-cann92-install`。
  原生产源码和 `.cache/csa/native-install` 保持原样，必须显式 `--cann92-abi` 才会加载实验产物。
  此构建及后续实测单独记结果，尚未将该实验当作官方支持组合。
- 仅修正RMS的实验已越过Q norm，随后在rotary发现同样的custom/内置ABI差异：
  仓库接口多了 `negate_sin`，参数错位使workspace报告不合理的174764 GiB，而非真实HBM不足。
  结果保留在 `native_forward_cann92_v2`。后续优先验证与原仓库C++封装匹配的custom bundle，
  不以持续删参代替配套算子，尤其QLIv2还有共享页stride及candidate参数。
- [100步六档汇总](results/local_20260921/native_trace_v2/native_groups_device1.json) 全部通过，
  每步均核对五组metadata和完整sentinel allocation；第25/75步交换请求位置。
  本轨迹每步重置sentinel，尚未验证完整CSA浮点state的跨步演进。

## 15. 21:03～21:05：配套custom包与真实DP2结果

五个算子 `compressor_metadata;rms_norm_dynamic_quant;inplace_partial_rotary_mul;scatter_nd_update_sk;quant_lightning_indexer_v2`
已从当前原始csrc构建、打包、安装成功。安装路径为任务目录 `.cache/csa/csa-native-ops-install`，
未覆盖旧全局vendor或原始C++扩展。完整构建/安装日志已经归档到 `results/local_20260921/logs/`。

使用原始C++扩展及该custom包执行 BS4/query6/128K 的真实单层原生forward，
已越过先前RMS/rotary接口点，但在旧全局vendor的Compressor tiling失败：
`state_cache must be contiguous, first axes stride should be equal to 4096, but got 8192`。
见 [最新原生结果](results/local_20260921/native_forward_custom_v3/native_forward_B4_L131072_device0.json)。
当前仓库Compressor源码已将条件改成stride不能小于连续值，并将实际stride传入kernel；
下一步应提供同源码配套Compressor/SAS等包，而不是更改原生cache布局掩盖问题。
用户随后决定迁移，本机没有继续构建这两个算子，也没有完成完整原生或PTO数值验收。

[dsv4_csa_dp_metadata.py](dsv4_csa_dp_metadata.py) 已在本机两张卡启动两个真实worker：
TP1、DP2、world HCCL，DP CPU group Gloo；保留DeepSeek MoE分类，实际skip-allreduce=False。
调用原生 `NPUModelRunner._sync_metadata_across_dp` 和 `CudagraphDispatcher`，
没有mock collective，也没有使用dense模型绕过原生同步。

| 本地BS对 | max token | 同步模式 | 非空rank probe |
| --- | ---: | --- | --- |
| 4 / 40 | 240 | FULL | 五组metadata、eager/graph、整个cache保护区通过 |
| 40 / 4 | 240 | FULL | 同上 |
| 8 / 24 | 144 | FULL | 同上 |
| 16 / 32 | 192 | FULL | 同上 |
| 0 / 4 | 24 | NONE | 非空rank通过；空rank只验协调 |
| 0 / 40 | 240 | NONE | 非空rank通过；空rank只验协调 |
| 4 / 40，rank0显式NONE | 240 | NONE | 验证全rank模式降级；独立probe仍自测graph |

每个case都分别验证 `allow_dp_padding=False/True` 的token向量。
实际capture sizes由原生配置解析为验6的倍数，结果JSON保留完整列表。
[双rank汇总](results/local_20260921/native_dp_v1/dp_metadata_summary.json) 为PASS，两个worker均正常退出。

范围限制：本fixture没有PD connector、没有MoE/EP计算、没有EPLB、没有完整CSA。
空rank没有执行实际model runner dummy attention；probe图也不等于完整模型图。
所以只能将P4的metadata部分记为通过，不能将整个P4或DP16场景记为通过。

## 16. 用户决定迁移后的归档

用户要求将全部工作上下文保存在当前仓库，以便在真实16卡环境下载并接续。
本机数值/环境修复工作到此收尾，已结束的DP2结果被纳入交接；未把剩余目标标为完成。

- 根目录 [DSV4_FLASH_CSA_HANDOFF.md](../../DSV4_FLASH_CSA_HANDOFF.md) 提供接手入口与新session提示。
- `handoff/` 保存后续16卡步骤、环境/构建说明、源码导航、精确版本、原有PyPTO差异、构建实验patch与脚本参数快照。
- `results/local_20260921/logs/` 保存被git忽略的关键日志；来源和SHA见 `handoff/log_archive.json`。
- 原始JSON、失败记录及同事HTML保留；状态汇总新增 `handoff_state.json`，不覆写历史证据。
- Git不携带权重、虚拟环境或本机构建二进制，已记录它们的来源、配置、哈希和重建条件。
- 新机器不得把旧compressed-tensors参考层替代目标Ascend/ModelSlim权重验收。

最终模型解释器的distribution metadata与选中的源码可能不同：例如metadata显示PyPTO0.2.1，
实际import来自锁定的 `pto_eager/pypto`；vLLM/Ascend同样通过源码选择器固定。
因此交接同时保留包版本、实际import路径和Git SHA，以后两者确定此次执行的源码。

## 17. 2026-09-21：16卡机器恢复环境（CANN 9.0，进行中）

用户要求先恢复本地工具链，再按原验证计划继续；PyPTO 和 Simpler 必须使用
`feat/kernel-mode-integration-test` 调试分支。当前环境不能按原机器 CANN 9.2 的组合照搬。

已核对与执行：

- 本机可见16个 Ascend910 逻辑设备，驱动26.0.rc1；CANN 安装信息为9.0.0。
  运行前通过 Simpler 的 `onboard-arch-precheck` 检查，平台为 `a2a3`。
- 默认解释器为 Python3.13，未装 Torch；现有共享 Python3.10 环境为
  Torch2.6.0/torch_npu2.6.0.post5。新建工作区 `.venv`，使用 Python3.10.19，
  带 `--system-site-packages`，禁用用户site，安装只写项目环境。
- 根据[昇腾官方安装说明](https://github.com/Ascend/pytorch/blob/master/README.zh.md)，
  安装与 CANN9.0 配套的 Torch2.10.0/torch_npu2.10.0.post2。
  实际导入版本为 `2.10.0+cpu` / `2.10.0.post2`；C++11 ABI为True，
  `_npu_shutdown_synchronize`、`_npu_shutdown` 均存在。
- 依赖下载最初遭遇代理403；直连PyPI/PyTorch下载较慢，停止本轮下载后改用清华镜像成功安装。
  构建依赖按 PyPTO 的 `build-constraints.txt` 固定，补齐其 kernel-mode CI 的 CANN Python依赖。
- PyPTO源码为 `02c0026993b08353e6cee4fdb93e86eddabb8701`，
  Simpler源码为 `e914837d540a899dfce0cf48f2adac91e3884930`；两个顶层仓库保留调试分支。
  PyPTO原有runtime pin为17ea300，与本地Simpler不同；已将runtime子模块及
  `SIMPLER_KERNEL_REVISION` 同步到e914837，适配器将从相同SDK源码编译。
- 本地 `env.sh` 设置 CANN9.0、GCC15.2、PTOAS0.61、`PTO_EAGER_ROOT`及项目venv。
  最初选择现有0.63，随后根据当前PyPTO `toolchain/versions.env`校正为0.61并核对二进制版本。
  系统 `pypto-setup` 指向不存在的ptoas-bin，因此本地明确使用实际PTOAS安装目录。
- Simpler已从本地调试分支完成editable安装，构建目标为`build_package_a2a3`，
  包括A2/A3 onboard runtime。PyPTO正在启用`PYPTO_BUILD_TORCH_NPU`及测试辅助模块编译。
  使用资源loader默认的2个PyPTO编译任务。
- 找到两份本机权重；`/data/model/dsv4-flash-0731-dspark-w8a8`的48个分片完整，
  其配置为compressed-tensors W8A8；用户后续明确仅作参考，最终权重改为BF16（见第18节）。

当前注意事项：本仓库要求Torch2.10，但锁定的torch_npu2.10.0.post4属于CANN9.1组合。
本轮采用CANN9.0的post2，尚不能据导入成功认定完整原生CSA可运行。
PyPTO原有退出清理版本门禁仍只接受2.6.0.post2，已新增2.10.0.post2测试参数，
待构建完成后先复现拒绝，再验证必要适配及硬件退出顺序。

原始安装/构建日志暂存`.cache/csa/setup-cann90/`；后续结果使用独立的
`results/cann90_20260921/`，保留原`results/local_20260921/`历史证据。
本节只记录恢复过程，P0完整原生基线、完整PTO CSA及P2～P5状态尚未改变。

## 18. 2026-09-21：用户更新最终权重及PD部署要求

- `/data/model/dsv4-flash-0731-dspark-w8a8`（48分片）仅作为参考。
- 最终目标使用BF16格式权重；其他人今晚下载，完成后用户告知路径。
  因此旧计划中最终 `quantization=ascend`/W8A8要求被用户明确替换，ND布局要求保留。
  目标checkpoint到达后重新核对配置、DSpark权重、各层dtype及cache方案；此前单层结果注明合成或参考。
- 当前没有可用的prefill服务；P5由本session先运行P，再运行D，完成真实PD链路验证。
  准备配置与资源安排可以先行，最终BF16模型验收需等待目标权重。
- 验证计划同步这些更新，旧机器结果和移交材料作为历史证据保留。

Simpler基础回归已完成：295 passed、1 skipped（未构建的sim runtime）、36 warnings，
范围为`test_task_interface.py`及`test_worker_kernel_mode.py`的非硬件用例。
原始日志：`.cache/csa/setup-cann90/simpler-unit.log`、`simpler-unit.xml`。

## 19. 2026-09-21：复核性能测试脚本的权重格式

用户随后表示BF16目标不确定，要求检查
`vllm-ascend-main/tests/dsv4_perf_accuracy_20260827`。只读检查得到：

- `runtime/config.sh:3`默认模型为`DeepSeek-V4-Flash-0731-w8a8-dspark-0819-new`，
  可由外部`MODEL_PATH`覆盖；该默认目录在当前机器不存在。
- `runtime/decode/run_dp_template.sh:71`及`runtime/prefill/run_dp_template.sh:73`
  均显式传入`--quantization ascend`，未指定`--dtype bfloat16`。
- Decode第38行`VLLM_ASCEND_ENABLE_NZ=2`的注释涉及保留BF16层的物理布局，
  不能解释为全模型BF16。脚本默认意图是W8A8量化模型，具体skip层须核对目标checkpoint。
- 现有48分片权重虽为compressed-tensors W8A8，尚无证据证明与上述目标checkpoint完全相同。
- 已向用户说明证据，并询问最终恢复W8A8还是维持BF16；在确认前继续与格式无关的环境恢复。

PyPTO安装完成；新增版本门禁测试先复现2例失败，再允许精确版本2.10.0.post2。
kernel-mode CI单元回归：611 passed、4 warnings，68.29秒。
真机命令第一次提交被task-submit关键词规则拒绝：测试文件名中的`shutdown`触发
系统关机命令过滤，未获得设备、未运行。后续提交使用可检查的测试入口，保留队列设备锁。
原生custom包首次下载受代理影响，移除代理后已通过下载和host预编译；
第二次在构建脚本缺少Python `regex`依赖处停止，模型依赖安装完成后重试。

## 20. 2026-09-21：最终目标恢复W8A8（用户明确确认）

用户回复：“那就不用BF16的，就用W8A8的”。本指令取代第18节的纯BF16权重变更。
验证计划已恢复W8A8/`quantization=ascend`目标；Native/PTO仍统一ND，保留模型指定的BF16例外层。
现有48分片权重先作参考验证；用户此前对其“仅作参考”的定位未被改写，
最终checkpoint路径、量化描述、DSpark配置及权重身份需要实查。
没有现成P服务、由本session先启动P再启动D的要求仍有效。

## 21. 2026-09-21：CANN9.0运行时和原生构建复核（进行中）

环境基础包已安装：Torch2.10.0+cpu、torch_npu2.10.0.post2、当前调试分支PyPTO/Simpler，
以及锁定vLLM `84030bbe3d74d99bad477a3d2e37a973ccd8865c` 和本仓库editable包。
pypto-lib选用 `205255b4770ee84dfa176bcbc7bbef651953c7e1`。
模型依赖安装中保留NumPy2.2.6以满足PyPTO；Triton-Ascend3.2.2的NumPy1.26.4声明
由显式override处理，仍需实际kernel验证。上游Triton与Ascend安装覆盖曾导致导入失败，
最后单独重装Triton-Ascend后导入恢复，实际backend仅为Ascend。
这类依赖覆盖及尚未解决的FastAPI版本约束不视为正式P5环境验收通过。

真机任务 `task_20260921_220229_2226205214` 使用device8和task-submit资源锁，
结果为18 passed、1 failed，278秒；日志见
[runtime/kernel-runtime.log](results/cann90_20260921/runtime/kernel-runtime.log)。
8个退出清理场景、3个eager场景、异步Torch队列、torch.ops compile及5个capture场景通过。
唯一失败为 `test_capture_entry_interop[1-build-dir-mixed]` 的持久化缓存计数断言。
在断言中增加stats诊断后单独重跑任务 `task_20260921_220917_24411411392`，
确认原因是系统 `/usr/local/ptoas/0.61/bin/ptoas` 包装脚本不属于PyPTO可审计的launcher语法，
产生2次cache bypass。没有放宽缓存断言；改为安装分支锁定的官方cp310 wheel并校验SHA256。

原生构建的当前进展：

- 本地补齐GNU patch2.8及CMake3.31.10，构建工具写入`.cache/csa/build-tools/`。
  PyPTO仍使用GCC15.2，CANN原生构建改用系统GCC10.3.1，避免旧Abseil与GCC15的头文件冲突。
- 原生扩展两次Ninja构建在CANN `extract_host_stub.py` 的对象路径查找失败。
  仓库setup.py明确仅支持Make生成器；切换Unix Makefiles后已越过该失败点，仍在编译。
  未修改系统CANN脚本或套用旧机器CANN9.2 ABI实验补丁。
- 当前QLIv2源码依赖CANN9.0没有的诊断宏。添加带`#ifndef`的兼容定义，保留参数检查及错误报告。
  同时处理op-common和opdev的同名OP_LOGE宏参数含义不同的问题；此项仍待完整host构建验证。
- 配套custom包选择Compressor、QLIv2、SAS及各自metadata、RMS动态量化、partial rotary、scatter共9项，
  为完整Native CSA恢复准备。当前尚未生成并验证可用包，不能宣称环境全部恢复。

新增可检查入口 [run_kernel_runtime_validation.sh](run_kernel_runtime_validation.sh)，
必须在task-submit分配后运行，并使用TASK_DEVICE，输出独立JUnit与NPU日志目录。
最初多行`--run`提交仅执行首行，没有运行测试，该退出码0不计作通过证据。

## 22. 2026-09-21：原生扩展恢复、Python3.10与参考权重兼容

原生C++扩展使用CMake3.31.10、GCC10.3.1、Unix Makefiles完成编译和隔离安装。
`.cache/csa/native-install/vllm_ascend_C.cpython-310-aarch64-linux-gnu.so`已在当前Torch/NPU组合下成功导入。
PyPTO/Simpler实际导入路径均为本工作区的调试分支源码。

修复并验证的具体差异：

- 当前scikit-build-core1.0.3的editable hook改名为`_editable_skbc_<name>.py`。
  测试选择器现在识别新旧两种命名，并要求恰好存在一种，仍拒绝混用来源。
- Ascend声明支持Python>=3.10，但indexer/cache dtype补丁使用3.11的starred subscript语法。
  改用`typing.Literal[existing_args + ("int8",)]`，保留原Literal成员与幂等性。
  3个回归通过；整个`vllm_ascend/`在Python3.10的compileall通过。
- CANN诊断宏兼容回归2项通过（旧SDK缺宏、后续opdev重定义OP_LOGE、新SDK已有宏保留），
  并在正常UT conftest下再次通过。custom包host编译已越过原错误，设备kernel生成仍在进行。
- 第一次真实单层加载在`wq_b.scale`失败；当前48分片参考权重与旧机器参考权重不是同一导出形式。
  依照`DeepseekV4ForCausalLM.load_weights`补齐`.scale → .weight_scale`映射后，
  第二次暴露`[out]`对`[out,1]`的形状差异。测试加载器只对该channel scale补单元素轴，
  不转置或重写权重值；记录checkpoint形状、加载形状、dtype与SHA256，仍做逐值相等检查。
  新旧两种scale格式CPU回归2项通过；第三次真机布局验证排队中。
  这是参考fixture的规范化，不能当成最终checkpoint原生整模型加载已通过。

Python3.10修复测试见`kv_dtype_python310.xml`，CANN诊断测试见`cann_error_log_full_conftest.xml`，
scale加载测试见`reference_weights.xml`，均位于`results/cann90_20260921/`。
新增`run_csa_validation.sh`封装单卡layout、metadata、native forward和metadata算子smoke，
所有硬件运行使用task-submit分配的TASK_DEVICE。

系统CI缓存的PTOAS wheel无读取权限，未更改其权限；继续从官方release下载。
下载较慢，确认服务器支持HTTP Range后对剩余内容分3段下载，合并后必须匹配PyPTO锁定的SHA256才能安装。
本节记录时仍未完成PTOAS缓存重验，也未将完整CSA/P2～P5状态标记为通过。

## 23. 2026-09-21：PTOAS与9算子包安装完成，等待真机队列

官方PTOAS0.61 cp310/aarch64 wheel下载完成，SHA256为
`f808968019a11598b9418bfb25e4fc81c6869e12683b67921f15743095ac8df5`，
与当前PyPTO `toolchain/versions.env`完全一致。安装在项目独立PTOAS venv，
根`env.sh`已切换到该目录；PyPTO launcher依赖盘点通过，见`ptoas_inventory.json`。
持久化缓存真机重验已提交`task_20260921_223617_292468917783`，未据CPU盘点通过替代真机结果。

配套custom包构建完成并安装于`.cache/csa/csa-native-ops-install/vendors/custom_transformer`。
未改动全局vendor。安装内容核对包含7个AICore算子、2个AICPU metadata算子，
9个ACLNN执行符号及9个workspace符号均可解析；`env.sh`加载此隔离vendor。
产物SHA及清单见`custom_ops_manifest.json`；目前状态为已构建安装、尚待设备执行验证。

资源状态：从22:26起，另一ci-runner任务占用全部16个设备。当前真实单层ND布局、
PTO缓存复验、原生metadata算子smoke都通过task-submit排队，没有绕过锁或终止其他任务。
当前包版本、源码SHA、参考权重config/index哈希及兼容约束见`environment_restore.json`；
已结束的关键日志复制到`results/cann90_20260921/logs/`，映射与SHA见`log_archive.json`。
完整PTO adapter、P2数值、完整P3/P4与P5仍未通过，继续按前置依赖顺序推进。

## 24. 2026-09-21：服务入口CPU检查与依赖约束复核

`python -m vllm.entrypoints.openai.api_server --help`退出码0。
使用锁定vLLM源码的`launchers.app.build_app`构建HTTP应用，TestClient访问
`/version`与`/openapi.json`均返回200，注册20个API路径，包含chat completions。
证据见`http_app_smoke.json`及归档日志；此检查不包含模型、引擎或PD传输。
首次检查使用旧版`openai.cli_args`导入路径失败，改用本提交的`launchers.cli_args`后通过。

补齐缺失的msgpack后，`uv pip check`仍报告3项已知版本约束冲突：
Triton-Ascend声明NumPy1.26.4而PyPTO要求NumPy>=2（当前2.2.6）；
本Ascend提交声明torch_npu2.10.0.post4而CANN9.0配套选择post2；
Ascend声明FastAPI<0.124而锁定vLLM要求>=0.133（当前0.136.3）。
这些约束未通过修改包元数据隐藏；HTTP检查只覆盖所述API行为，完整运行兼容性仍待P2～P5验证。

22:48设备仍由同一CI任务占用。layout的客户端等待超时，任务
`task_20260921_222800_267171222322`在分配设备前被task-submit自动取消，未运行。
确认其状态为not_found且日志明确记录取消后，重新提交异步任务，避免等待超时再次取消排队。

## 25. 2026-09-21：当前分支probe前端复核与队列入口

三个现有probe（五组metadata读取、C4紧凑slot展开、共享INT8 K/FP16 scale页）
在当前PyPTO调试分支完成CPU前端解析、Torch schema注册及A2/A3代码生成，均通过。
证据见`probe_registration_cpu.json`与`probe_compile_cpu.json`。
此步骤产生PTO/C++代码，不包含设备binary执行或数值验证。
metadata probe保留编译器的`ScalarWriteLineShared`保守警告：循环下标为运行时值，
编译器无法证明不共享缓存行；本probe每worker写整行16个INT64（128字节），
不同worker按token行分工，未关闭诊断或改变原有真机断言。

布局重提任务为`task_20260921_224958_341820230163`。
P1的B4、100步接受轨迹复验独立排队：`task_20260921_225140_345224429421`；
当前未分配设备，未把旧机器PASS计入本次结果。
`run_csa_validation.sh`新增`dp_metadata`入口，必须由队列分配两卡，
通过原生支持的`ASCEND_RT_VISIBLE_DEVICES`映射到子进程逻辑卡0、1。
脚本语法检查通过；DP2执行将在单卡复验通过后提交。

## 26. 2026-09-21 23:00：真机验证待资源，保留异步任务

另一CI任务`task_20260921_222643_245925220139`仍占用全部16张卡，
从22:26持续至本节记录时间。已向用户说明并询问预计释放时间；未绕过队列或干预该任务。
为避免原`--run --timeout`客户端再次取消排队，将尚未运行的缓存复验和算子smoke任务
明确取消后重新异步提交。没有取消运行中的设备任务，没有重复运行结果。

当前待验队列（实时状态以`task-submit --status <ID>`为准）：

| 任务 | ID |
| --- | --- |
| 参考权重单层ND/layout | `task_20260921_224958_341820230163` |
| P1 B4/100步metadata与graph | `task_20260921_225140_345224429421` |
| Native CSA B4/131072历史预执行 | `task_20260921_225912_378374631592` |
| 官方PTOAS的缓存失败用例重验 | `task_20260921_225920_378690623946` |
| Compressor/QLI/SAS metadata算子smoke | `task_20260921_225938_379430129650` |

已取消的旧排队ID为`task_20260921_223617_292468917783`与
`task_20260921_223801_302349831531`。上述Native预执行会重新严格加载参考单层权重，
使用原生metadata和完整历史；只检查原生执行及输出有限性，不计为P2 Native/PTO数值对照。

本次环境复现说明见[ENVIRONMENT_CANN90.md](handoff/ENVIRONMENT_CANN90.md)。
下一步先检查上述原始结果；通过后扩展P1六档、双卡metadata，并继续完整PTO adapter与数值对照。
当前不能宣称环境真机验收或完整CSA接入已完成。P5仍需要前置阶段、最终W8A8 checkpoint和P/D资源。

## 27. 2026-09-21 23:11～23:20：设备释放后的实际结果

前一轮为有效进展：完成环境修复、CPU证据及真实任务提交；本轮重新查询队列，
确认占卡CI和5项待验任务均已结束，再读取实际JSON/JUnit及任务日志。

| 检查 | 本机结果与证据 |
| --- | --- |
| 第2层参考W8A8加载及ND/five-cache布局 | PASS，`native_layout_v3/native_layer_layout_device0.json` |
| Compressor/QLI/SAS metadata算子 | PASS，`native_ops/`下3个报告，仅算子可执行性 |
| B4、100步metadata/graph | PASS，`metadata_trace_b4/` |
| B8/16/24/32/40、各100步metadata/graph | PASS，任务`task_20260921_231323_5977593150`，`metadata_trace_remaining/` |
| 两个真实DP rank、不均衡/空rank协调 | PASS，任务`task_20260921_231528_62457128435`，`dp2_metadata_v2/` |
| B4/131072历史Native完整attention | FAIL，执行完成但输出含非有限值，`native_forward_b4/` |

双卡首提任务`task_20260921_231323_5978296288`退出2：task-submit自动追加
`--device 9,10`，原DP脚本不接受此参数。脚本现在接收并核对它必须等于wrapper设置的
两卡可见性mask；复验使用真实两卡通过，未固定或绕用未分配的物理设备。
该PASS仍限于DP2 metadata、同步和probe graph，不覆盖完整CSA/dummy/EP16/PD。

上述metadata旧断言未覆盖每个builder实际传给QLI的压缩长度缓存；第29节针对完整forward
暴露的新缺陷增补此检查，不能用本节PASS替代新增断言的复验。

## 28. 2026-09-21：持久缓存恢复与双目录editable依赖盘点

官方PTOAS重验任务`task_20260921_225920_378690623946`仍失败，原因已经变化：
`pypto.pypto_core`实际位于editable安装的第二个包搜索目录，旧`_package`仅盘点
`__file__`所在的源码目录，因此报告external import redirect。CPU打印实际`__path__`
确认源码和本工作区venv扩展目录都由安装器明确声明。

修复PyPTO依赖盘点：枚举全部声明的package搜索根，连同各根原生扩展的ELF依赖一起哈希；
已加载模块若在这些根之外，仍拒绝缓存。新回归先失败，修复后identity/inventory/artifact
CPU回归163项通过，见`cache_identity_cpu.xml`。EN/ZH缓存文档已同步。
真机任务`task_20260921_231815_66246726584`重跑原失败用例，1 passed、18 deselected，65.09秒。
加上此前18项runtime通过证据，选定19个场景现都有通过结果；不是单次19项重跑的计数。
没有放宽缓存断言、关闭缓存或更换非调试分支。

## 29. 2026-09-21：Native非有限值的阶段诊断（进行中）

新增测试专用`--diagnostics`，对关键阶段输出记录dtype/shape/有限性及范围；
诊断含同步，只用于定位，不能作为性能或异步流正确性证据。
任务`task_20260921_231726_65477725579`表明QKV、RoPE输入、两个compressor均有限，
indexer输出全为-1，sparse attention输出458752个非有限值，之后传播至O-proj。
结果见`native_forward_b4_diagnostic/native_stage_trace.json`。

定位到`AscendDSAMetadataBuilder._build_qli_metadata`：复用其他builder生成的
调度metadata时，当前builder自己的`qli_seqused_k`和`qli_cmp_residual_k`未刷新，
实际indexer因此可能读取初始化的0长度。两builder共享metadata、连续两步的CPU回归
先复现2项失败（INT32/INT64输入）；已将当前builder长度缓冲刷新移到缓存命中判断之外，
保留调度metadata共享及稳定地址。当前正在跑DSA单元回归和完整Native复验。
五组metadata probe也新增每个C4 builder的QLI长度/余数独立检查，随后重新覆盖graph与DP2。

DSA完整CPU测试文件最终55 passed、18 warnings，见`dsa_metadata_cpu.xml`。
本次修改的Python lint/format及两仓库diff检查通过。已归档完成任务和失败诊断日志，
更新环境manifest及当前验证计划；旧的PASS仍保留其原覆盖范围。

修复后的真机任务已异步提交：Native B4/131072为`task_20260921_232451_85447219488`；
六档各100步、含QLI长度新断言为`task_20260921_232451_85455823450`；
真实DP2、含QLI新断言为`task_20260921_232451_8546259439`。
23:29复核时占卡CI已完成，Native B4/131072和DP2修复复验都已PASS，
六档metadata已完成B4/8/16/24/32，B40仍在执行。
Native无诊断同步路径输出`[24,4096]` BF16且全部有限，abs max=1.25，
峰值NPU allocated为1259987456字节，证据在`native_forward_qli_fix/`。
这证明该参考fixture的原生非有限值已由QLI长度刷新修复消除，仍不代表Native/PTO数值对照通过。

## 30. 2026-09-21 23:31：QLI修复复验完成与权重来源澄清

四个异步任务均已实际完成且退出0，原始日志归档到`cann90_20260921/logs/`并记录SHA256：

| 项目 | 结果与证据 |
| --- | --- |
| Native B4、历史131072、seed1024 | PASS：`[24,4096]` BF16全部有限，abs max=1.25；`native_forward_qli_fix/` |
| Native B40、历史131073、seed1024 | PASS：`[240,4096]` BF16全部有限，abs max=1.359375；`native_forward_b40_boundary/` |
| B4/8/16/24/32/40各100步metadata | PASS：包含每个C4 builder的QLI压缩长度/余数新断言；`metadata_qli_fix/` |
| 两个真实DP rank的新QLI断言 | PASS：`dp2_qli_fix/`；仍限metadata/probe/协调 |

B40任务ID为`task_20260921_233013_93997711300`，峰值NPU allocated为6118369280字节。
上述两个Native结果只证明参考fixture完整forward可执行且输出有限，不是P2的18组Native/PTO数值对照。
P1 metadata记为PASS；P0目标权重/逐Tensor标准、P2/P3/P4完整CSA及P5仍未完成。

用户进一步确认：`/data/model/dsv4-flash-0731-dspark-w8a8`是**cann_recipe量化**，
48分片继续仅作参考；**vllm_ascend W8A8**目标权重下载好后将通知本session。
当前参考fixture按compressed-tensors描述读取并适配单层scale命名/shape，这不是格式转换，
也不证明cann_recipe权重可由正式Ascend整模型加载路径直接接受。
保持最终W8A8、Native/PTO ND、用户提供目标权重后重新核对正式加载与量化描述的要求。
后续先推进不依赖目标checkpoint的PTO计算与阶段对照，不将当前参考结果升级为目标验收。

## 31. 2026-09-21 23:46起：不依赖checkpoint的QKV阶段实现

新增`dsv4_csa_qkv_kernel.py`与`dsv4_csa_norm_rope_kernel.py`，实现BF16 ND投影、
Q LoRA RMS动态INT8量化、W8A8投影以及内部Q/K norm和interleaved FP32 RoPE。
先保留Native在投影、norm、RoPE之间的BF16舍入；没有直接套用参考库更宽的融合中间精度。
这些是测试目录中的阶段计算核，尚未接到生产`attention.forward`，不包含cache更新和CSA后半段。

新增`dsv4_csa_projection_compare.py`和队列wrapper的`projection`阶段，使用seed1024合成输入，
不加载任何checkpoint。每个阶段独立对照，并记录QA→QR→QB→norm/RoPE串联误差。
执行前写入JSON的门槛为：BF16 atol/rtol=1e-2，INT8逐元素一致，FP32 token scale
atol=1e-6/rtol=1e-5；未因失败调整这些门槛。显式关闭内部格式，检查输入/权重/输出使用
torch_npu base format（ND=2或NCHW=0），拒绝NZ；后者同PyPTO Torch互操作的base-format定义。

首提`task_20260921_234601_13385303688`在RMS量化小核的PTOAS编译失败：
4个FP32元素组成的归约缓冲仅16字节，不满足32字节对齐。改为8行物理tile并保留实际valid shape。
第二提`task_20260921_234914_13859397988`的投影/量化已执行，随后RoPE编译失败，
不能据此前没有异常认定投影数值通过。已改进报告，使后续阶段失败时仍保存前面已计算的比较指标。

RoPE失败已缩小到PyPTO生成的`pto.subview`：源tile的有效行数是runtime值，IR已推导出
子视图有效范围，但代码生成只写dynamic结果类型，没有输出`valid [...]`操作数，PTOAS因此拒绝。
新增PYPTO/PTOAS两种memory planner的最小CPU回归，修复前2项均失败；正在修复codegen并重建本地调试分支。
没有改动PTOAS版本、取消尾部有效范围或放宽数值标准以绕过该错误。

## 32. 2026-09-22：终止独立QKV路线，明确参考使用边界

用户指出计算参考应为pypto-lib的`decode_csa_tp1`，并要求接入时参考main与Qwen两个
已有仓库；随后再次明确“只是参考，不是照抄”。此前自行重写QKV并追其编译问题偏离主线。

三份独立QKV/RoPE实验文件已移到`experiments/retired_qkv_20260921/`，
`run_csa_validation.sh`删除`projection`入口。两次失败结果保留，不计为数值通过。
补记第31节最终结果：inherited-subview codegen CPU回归223 passed；无该修复的NPU复验，
不再沿独立实验路线继续调试。

读取核对main的weights/cache/backend/dispatch及Qwen的attention替换/custom op/warmup。
main当前HEAD为`4a8bb47`，Qwen为`093b1eb`。确认旧main存在S=8、每轮固定两条压缩写入，
且其量化ABI是Q-A/KV INT8、O-B BF16；这与当前参考函数Q-A/KV BF16、O-B INT8不同。
Qwen又有BF16、无padding等自身限制。因此这些文件只用于核对方法和约束，未复制其代码。
计划中已补入对照位置、差异及执行顺序；最终量化拓扑仍等待目标checkpoint实查。

## 33. 2026-09-22：提取参考TP1的完整attention core

在pypto-lib原`decode_csa.py`中提取`_decode_csa_tp1_attention`，输入/输出均为
`[T,4096]` BF16，移出HC参数、外层norm及HC输出。原`_decode_csa_tp1`保留HC壳并调用同一core。
QKV、原始KV写回、两个compressor、indexer、sparse attention及TP1 O-proj保持原计算代码和依赖。
没有复制main/Qwen计算实现，也没有加入独立QKV实验。

源码核对`source_audit.json`：移出HC部分后，attention计算体AST与固定基线完全一致；
六个被调用的计算模块与基线逐字节一致，并记录SHA256。

CPU完整pass lowering：attention-only入口及原HC入口均PASS，使用原参考TP1/S=8配置；
证据在`results/cann90_20260921/reference_attention_extract/report.json`、`lower.log`及两份IR。
本项不调用NPU、不含PTOAS生成/执行，也不代表S=6、Native物理cache或P2数值已通过。
接下来处理显式TP1/S=6配置和Native cache适配，再运行完整B4对照。

## 34. 2026-09-22：纠正代码归属并撤回非必要依赖改动

用户明确指出CSA代码必须直接写在`vllm-ascend-dsv4-pto`，并质疑PyPTO修改范围。
第33节把提取写进pypto-lib属于错误实施位置。本轮已归档该差异后撤回，
`pypto-lib`当前HEAD仍为`205255b4770ee84dfa176bcbc7bbef651953c7e1`，工作区干净。
本轮曾尝试向参考仓库增加S=6改动，但patch校验失败且未写入；随后所有实现改在本仓库。

CSA core及TP1实际调用到的子函数已提取至
`vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/`；`REFERENCE.json`记录固定来源及函数清单。
不包含HC/TP通信/独立golden入口，也未复制main或Qwen的adapter。
使用包内相对导入和显式TP1/S=6常量，不读取或修改服务`sys.argv`。

PyPTO改动逐项核对后：

| 类别 | 真实原因/证据 | 当前处理 |
| --- | --- | --- |
| Simpler pin与`SIMPLER_KERNEL_REVISION` | 分支原pin为17ea300，本地调试分支为e914837；需使用同一SDK编译运行 | 保留e914837对齐，顶层PyPTO/Simpler仍为指定调试分支 |
| TorchNPU退出版本门禁 | 原代码仅接受2.6.0.post2；本机按CANN9.0使用2.10.0.post2；611项CPU回归和8个真机退出场景已有证据 | 保留精确2.10.0.post2兼容及对应测试/文档 |
| 双目录editable缓存识别 | 合法扩展目录被判为external redirect，触发cache bypass；不是首次CSA执行的硬阻塞 | 撤回代码、测试及EN/ZH文档改动 |
| inherited-subview codegen | 独立QKV实验触发；223项CPU通过，但无NPU复验 | 撤回C++及回归测试改动 |
| cache ST诊断文字 | 缓存失败时打印stats | 随缓存改动撤回 |

撤回diff保存于`handoff/patches/reverted-pypto-cache-and-subview.patch`及
`reverted-pypto-lib-attention-extraction.patch`。PyPTO已用2并发重新编译并安装；
构建产物与实际安装`pypto_core`的SHA256一致，为
`d8479cb856f7568d58fd45b3b01572c482d8f5947c1806dc7b70aa16848b6f6f`。
第28节持久缓存PASS对应撤回前配置，不能继续称为当前无补丁缓存复验通过；
先前各次实际测试结果仍作为历史证据保留。

## 35. 2026-09-22：目标仓库中的TP1/S=6完整链路CPU lowering

配置容量使用B64/S6（384行，满足参考O-proj的128行tile）覆盖目标实际B4～40。
修复本仓库Indexer compressor的压缩行编号：每请求预留`ceil(S/4)`行，
根据device起始位置定位实际闭合token；未用行初始化，写回严格限制在该请求的6个token内。
池化、归一化、Hadamard和量化公式继续使用参考计算。

首次包导入暴露一个从独立参考入口带来的未用prefill import，已移除。
第二次lowering发现新增初始化scope与后续scope复用`request`名字引起scalar跨scope，
已在本仓库改为局部独立命名`init_request`；没有修改PyPTO绕过它。
最终完整链路CPU lowering PASS，验证导入前后argv不变；证据为
`results/cann90_20260921/csa_integration_s6_v3/report.json`，原始日志为
`csa_integration_s6/lower_v3.log`。新增可复现入口`dsv4_csa_reference_lower.py`。
Ruff检查与format通过。

本项仍不是NPU数值验收：Native cache物理布局、FP16 indexer scale共享storage、
state搬运及完整B4 Native/PTO对照尚待接入；P2～P5仍未完成。

## 36. 2026-09-22：用户指定正式目标权重，下载尚未完成

用户纠正目标路径为`/data/model/DeepSeek-V4-Flash-0731-w8a8`，允许下载完成后使用。
此前检查的两个小写目录并非本次新下载的目标：48分片为已知cann_recipe参考，
46分片版本同样声明compressed-tensors，只有mtp.0且没有DSpark专用权重。

新目录已出现`quant_model_description.json`，声明`model_quant_type=W8A8_DYNAMIC`；
第2层QA/KV/OA和两个compressor为FLOAT，QB/Indexer QB/OB为W8A8_DYNAMIC，
与当前参考TP1计算链的量化拓扑相符。配置含`dspark_block_size=5`及target层40/41/42，
量化描述含main_proj、markov_head和confidence_head。下载文件名标示共75分片，
首次检查存在多个`.incomplete`，第二次检查仅12个完成分片，尚未获得完整索引。
这些是下载中状态，不能据此宣称目标权重完整或原生加载成功。

继续推进不依赖checkpoint的Native存储适配。当前新增改动仍处于开发验证阶段：
Indexer共享页改为单一可写存储参数；两个compressor norm保留Native FP32；
新增device metadata及每请求14行state搬运kernel。完整链路CPU lowering仍在排错，
第35节PASS只对应当时版本，不代表上述未完成改动已经通过。

## 37. 2026-09-22：Native存储适配真机验证与完整B4对照入口

所有新增代码仍在本仓库，PyPTO/Simpler及pypto-lib未增加修改。
Indexer通过一个FP16载体参数描述整页共享存储，内部将K区域reinterpret为INT8，
scale直接按Native FP16读取/写回；动态页stride保留，未复制完整历史cache。
Indexer读取阶段使用只读参数，写回阶段由外层唯一InOut参数声明变更。
主/inner compressor norm保持Native FP32，main/inner压缩RoPE各取本组metadata。

`native_metadata.py`提供实际device lengths/bounds/slots、RoPE布局转换及14行state窗口搬运。
首次CPU lowering暴露动态reinterpret的整除约束和混合scalar/tensor循环作用域问题；
分别通过FP16存储载体、分开slot计算与RoPE搬运处理，未修改依赖编译器。
五个adapter kernel和完整计算链均完成CPU lowering；最新完整链路为`csa_native_storage_lower_v6/`。

首轮NPU任务因未显式返回外部Out参数，在Torch注册阶段失败，没有执行kernel；
补齐返回声明后，两卡并行和后续最大batch边界验证通过：

| 用例 | 任务ID | 结果 |
| --- | --- | --- |
| B4 / 131072 | `task_20260922_092557_89094120616` | PASS，`native_adapters_b4_v2/` |
| B8 / 131073 | `task_20260922_092558_89283924951` | PASS，`native_adapters_b8_v2/` |
| B40 / 131071 | `task_20260922_092841_120649123575` | PASS，`native_adapters_b40_boundary/` |

上述测试使用真实Native五组builder/BlockTable/DeviceMetadataExecutor、实际页stride和非零offset，
逐元素检查位置、SWA页号、main/inner紧凑slot与RoPE；两个state的历史搬运和当前行写回
与Native逻辑view对照，整份allocation比较包含padding/未写区域/保护区。
它们不读取参考checkpoint权重payload，也不计作P2完整attention或目标权重验收。

新增`native_adapter.py`和`dsv4_csa_full_compare.py`，准备已加载ND权重，串接同一参考完整链，
比较输出、六个cache/state、Indexer Top-K和未写区域。阈值先于执行写入JSON，失败不放宽。
当前先用已知参考单层调通，不使用未下载完成的目标，也不将参考loader应用到ModelSlim目录。
首个完整B4任务`task_20260922_093431_273708411819`完成Native后，在适配器误把RopeDataProxy
当字典枚举处失败；已改为显式传入本层名称，与Native相同方式读取。
复验任务`task_20260922_093551_279389929167`已提交，P2尚未通过。

## 38. 2026-09-22：完整B4编译失败与Native未写区域异常

第37节后续任务均已结束，尚未执行完整PTO attention 数学计算：

| 版本 | 任务ID | 失败点 |
| --- | --- | --- |
| v2 | `task_20260922_093551_279389929167` | Orchestration runtime Tensor不支持改变dtype；展平动态维乘积也无法由wrapper反推 |
| v3 | `task_20260922_093826_294622431278` | reinterpret移入InCore后，gather_row的源被lower为Tile，必须为GM Tensor |
| v4 | `task_20260922_094029_31775869998` | L1 FP16 tile不能reinterpret成INT8，要求flat/none_box布局 |

原始结果保存在`full_compare_b4_v2/`、`full_compare_b4_v3/`、`full_compare_b4_v4/`。
因此`csa_native_storage_lower_v6/`仅为当时版本CPU lowering通过，不能作为当前完整编译通过证据。
本轮改用单个INT8物理页载体：K按原生INT8直接读取，只将每页64字节scale区域在Vector中
reinterpret为FP16；scale写回由单任务合并以保留同页其他行。该方案仍需编译和真机验证。
这些调整仅在本仓库进行，未修改PyPTO/Simpler或pypto-lib。

完整对照还发现Native自身的未写区域检查失败：v4中compressed allocation前128字节保护区变化，
indexer allocation保护区有66字节变化；SWA及两个state的未写区域检查通过。
有效compact slots为`[3,0]、[1030,0]、[2057,0]、[3084,0]`。
具体偏移、变更量与完整异常保存在各次`full_compare.json`，原因待定位；不移除保护区、不放宽阈值。
该异常与PTO编译失败分别记录，均不能计作P2通过。

## 39. 2026-09-22：完整PTO首次真机执行，数值未通过；定位Native负slot写入

单个INT8页载体方案完成编译。v5的Vector assemble触发PTOAS tmov shape限制，改为将各页
64字节scale直接gather到INT8 Vector，再整体reinterpret FP16；没有改依赖编译器。
v6任务`task_20260922_094946_383250212736`完整执行Native和PTO，结果位于
`full_compare_b4_v6/`，状态仍为FAIL：

- SWA、主compressed KV及两个FP32 state逐Tensor容差检查通过。
- PTO全部allocation的未写区域/保护区检查通过。
- Indexer INT8 K有44个元素相差1，FP16 scale有1个元素超差。
- Top-K逐位置11324/12288不同；每行集合交集499～506/512，不只是排序差异。
- 最终输出无非有限值，但32555/98304个元素超差，max_abs=0.0655518，RMSE=0.0124437。
  阈值未放宽，P2未通过；后续需要逐阶段核对Native量化/舍入顺序。

Native scatter独立非零offset诊断任务`task_20260922_095044_392660329400`通过6组
BF16/INT8/FP16、offset组合，结果为`native_scatter_offset_v1/scatter_offset.json`。
完整Native诊断任务`task_20260922_095419_36140514356`随后确认：有效4个compact slot之后，
另有6个`[-1,31]`padding slot。其linear index为-1；`scatter_nd_update_hp.h`的batch/slice
快速分支均未跳过负索引，而no-sort分支已有范围检查。这解释了实际保护区写入，
不能用去掉guard或忽略尾slot掩盖。

本仓库对上述两个快速分支补充负linear index跳过；正索引的读取/写回路径不变。
独立回归扩展为有效slot之间及尾部夹入`[-1,31]`，使用至少一整页的前保护区，检查整份allocation。
当前正在重建本地custom包，尚未记录修复后通过结果。诊断数据保存在`native_full_diagnostic_v1/`。

安装前逐文件比对发现首次增量构建未更新生成目录中的kernel头文件，`cmp`阻止了安装。
该失败返回后误提交的完整任务`task_20260922_095926_71395317161`已终止，
`full_compare_b4_v7/`只作中止任务记录，不计验收。
同期旧二进制运行的负slot回归`task_20260922_095926_7147635551`得到稳定红例：
每个offset组合分别有1024个BF16 cache字节、128个INT8 key字节、2个scale字节在合法区域外变化。
证据为`native_scatter_negative_v2/`，不能称为修复后结果。
已仅作废scatter生成目录的17个构建stamp，再次编译并核对生成头文件与源码相同。

重建及安装成功，产物hash单独保存在`scatter_negative_fix_manifest.json`，
构建/安装日志为`logs/native-scatter-negative-build-v2.log.txt`和`logs/native-scatter-negative-install.log.txt`。
安装后任务`task_20260922_100345_103992630299`完成8组负slot/非零offset回归全部PASS，
新增256KiB行覆盖快速分支的slice路径；整份allocation逐字节比较，未放宽保护区检查。
结果为`native_scatter_negative_v3/scatter_offset.json`。

完整B4复验`task_20260922_100345_1041019215`（`full_compare_b4_v8/`）确认Native五组
allocation的未写区域检查全部恢复PASS；数值差异与v6相同，因此负slot写入不是这些数值差异的来源。
诊断入口复用`q_proj_qr`、`indexer_qr_rope`、`indexer_qr_hadamard_mm`，未另写QKV计算。
QR INT8有1568/24576个元素相差1；给定完全相同的Native QR及FP32 RoPE后，
Indexer Hadamard前BF16仍有31711/196608个元素不同，量化query有23495个元素不同。
Native/PTO中间值分别保存在`native_debug.pt`、`pto_debug.pt`。

源码核对发现具体舍入边界不同：Native QA输出先BF16再RMSNorm，Indexer QB反量化先BF16再RoPE，
Hadamard先以未缩放矩阵输出BF16再做缩放，QLI消费FP16 query scale与weights。
当前在本仓库保留参考函数链，补齐这些Native精度边界；不改参考仓库、不放宽数值阈值。
v9正在复验，上述精度适配尚未宣称通过。

v9任务`task_20260922_100728_16017379520`结束：QR INT8逐元素一致；给定同一Native QR及
FP32 RoPE的Indexer Hadamard前BF16、量化query及FP16 query scale也全部逐元素一致。
QR FP32 scale有9个元素不同，max_abs仅`1.1641532e-10`，诊断仍按零容差如实记FAIL。
完整链路仍FAIL：输出超差16356/98304、RMSE=0.00867586；Indexer K差22个元素，
scale已完全一致；Top-K逐位置10510个不同。所有未写区域检查继续PASS。

完整链路的metadata适配此前将Native FP32 RoPE降为BF16，独立诊断已证明保留FP32时
Indexer query边界可一致。现将RoPE参数在本仓库完整链及metadata适配中保留FP32，
只转换half/interleaved排列，移除冗余同dtype cast；对应metadata测试改为逐元素核对原生FP32系数。
CPU lowering通过（`csa_native_fp32_rope_lower/`），NPU完整B4与B8/B40 adapter回归已提交，尚未宣称通过。

FP32 RoPE回归结果：

- B8/131073 adapter：`task_20260922_101219_26614041675` PASS。
- B40/131071 adapter：`task_20260922_101219_266126316666` PASS。
- 完整B4 v10：`task_20260922_101219_266120219471`完成，仍FAIL。
  主compressed KV、Indexer INT8 K及FP16 scale均逐元素一致；两个state、SWA及所有未写区域通过。
  输出超差降至1613/98304、RMSE=0.00467185，Top-K逐位置仍有3198个差异。

进一步核对本仓库Native QLI源码，`ProcessVec0`以FP16计算weights×query_scale；
`FixpSToL1`以DEQF16和`0x3a800000`（1/1024）对ReLU(QK)舍入，再做加权归约。
参考链原来在这些位置保持FP32。在本仓库补齐该A3精度边界，保留原页表/候选/Top-K函数链。
分数跟随Native的1/1024单位，尚需v11真机验证；验收阈值不变。

## 40. 2026-09-22：QLI归约与O-proj原生量化边界

v11 `task_20260922_101445_285956413711`完成：24行Top-K候选集合均为512/512一致，
但第12行两对相邻元素顺序不同（202/203、470/471），PTO分数分别相差约`1.7e-10`和`1.2e-10`。
仍按原冻结的逐位置规则记FAIL，没有放宽为只比较集合。
最终输出还有722个元素超差，RMSE=0.00415008。

源码核对Native `_apply_output_projection`：WO-A输出BF16后拼接为8192列，WO-B按整个token
做动态量化；参考TP1原版对8个group各自量化。在本仓库保留原分组matmul tile及INT32 partial，
改为全部WO-A完成后统计同一token的amax，统一量化，并在INT32中合并各group partial后反量化。
首轮lowering报一个不同shape复用变量名，改为`output_token_scale`后重新验证，未修改编译器。
v12 `task_20260922_101935_104967331825`完成，输出超差降至7个、RMSE=0.00257040；
其余cache/state及未写区域通过；Top-K仍是上述4个位置，P2仍FAIL。

Native QLI的第二次matmul用FP16 score/coefficient做FP32 Cube归约，参考实现用Vector col_sum，
具有不同末位舍入。现通过现有`aiv_shard`/`aic_gather`接口保持有界tile，将此归约匹配Native Cube；
同时补齐主Q反量化→RMSNorm→RoPE之间Native BF16中间结果，尚需lowering和NPU数值验证。
所有改动仅位于本仓库；未变更PyPTO、Simpler或pypto-lib。

Cube归约尝试先在CPU lowering发现同一cross-core pipe混用NONE/LEFT_RIGHT；
将coefficient按UP_DOWN分片后lowering通过，但v13任务`task_20260922_102333_217604910696`
真机严重失败：Top-K 12287/12288不同，输出RMSE=0.0834037。该归约改动已撤回，
失败版本diff保存在`full_compare_b4_v13/cube_reduction_attempt.patch`，不能作为正确实现。
主Q的BF16边界改动保留并单独复验v14；生产链恢复已验过的Vector归约，4个Top-K顺序差异仍待解决。

## 41. 2026-09-22：输出通过；继续定位Top-K归约

用户明确目标权重暂时无法下载完，下载完成后会通知。本阶段继续执行不依赖目标checkpoint的
参考单层、metadata和graph工作，不再主动轮询下载或把参考权重当作目标验收。

v14 `task_20260922_102538_304201316532`结束：主Q的Native BF16边界补齐后，
输出仍有3/98304个元素超差，max_abs=0.0126953125、RMSE=0.0024109464；
Top-K仍为4个位置，六个cache/state及Native/PTO未写区域全部通过。

继续核对Native `_qkv_proj_rope`和attention executor：WKV输出、KV RMSNorm输出、
sparse attention输出均先发布BF16，后者再做inverse RoPE。参考融合实现此前省略这些中间舍入。
在本仓库函数链补齐对应边界后，v15 `task_20260922_103258_39899679990`执行完成：

- 最终输出PASS：超差0/98304，max_abs=0.0087890625、RMSE=0.0016622067。
- SWA、主compressed KV、Indexer INT8 K和FP16 scale逐元素完全一致；两个FP32 state满足冻结容差。
- Native/PTO五组allocation的未写区域全部PASS。
- Top-K仍有4个顺序差异，整体仍为FAIL，P2尚未通过。

v13失败产物显示coefficient经单槽cross-core pipe传到L1后，跨整个score循环持有；
同一pipe随后又传score tile，存在L1槽复用覆盖问题。新版本将每个query的FP16 coefficient
预先放入有界GM scratch（最大384×16×64×2字节），Cube直接加载到独立L1；
只保留score的Cube→Vector→Cube流转。CPU lowering通过
`csa_native_cube_gm_coeff_lower/report.json`，v16真机验证中，尚未宣称该改动正确。

v16 `task_20260922_103547_3207927224`结束：输出/cache/state/保护区继续通过；
Top-K仅两对顺序不同，且这次PTO保存的每对分数逐bit相等：
row12的1803/26567为`0.0007468020194210112`，
row14的15582/16862为`0.0007151686004363`。
核对Native `quant_lightning_indexer_v2_vector.h::MergeSort`，输入顺序为新块`mrgSrc`
在前、累计结果`mrgDst`在后；参考forest原来反向。按该Native顺序调整query merge操作数，
未修改分数、容差或逐位置比较规则。

v17 `task_20260922_103848_11740920093`完成exit=0：B4/S6/131072、seed1024完整对照PASS，
Top-K 12288个位置完全相同；输出max_abs=0.0087890625、RMSE=0.0016622067；
六个cache/state及全部未写区域通过。结果为`full_compare_b4_v17/full_compare.json`。
据此开始六档BS×三个长度的参考权重矩阵，其余17组经task-submit提交；
源文件hash和任务编号保存在`reference_matrix_v1/manifest.json`。
该矩阵属于参考层验证，不能替代目标ModelSlim checkpoint验收。

## 42. 2026-09-22：参考矩阵首次执行与scratch作用域

`reference_matrix_v1`全部任务已结束：B4三个长度PASS，其余15组在PTO core执行中FAIL，
未产生完整数值结果。`summary.json`逐项记录task状态；不是15组数值超差。
B8/131072 device日志定位到`FATAL: Task Allocator Deadlock - Heap Exhausted! ring=1`：
ring容量268435456字节，已用263748608字节，下一次申请6291456字节；
最老task属于尚未关闭的外层scope。B40同样耗尽256MiB。

当前`pypto.torch.init`的kernel接口未暴露heap容量设置，不通过私有字段或依赖仓库补丁扩大。
根CSA的外层scope原先把Indexer的大型Top-K scratch与后续attention/O-proj同时保留。
在本仓库为完整Indexer函数调用增加独立`pl.scope()`，输出仍写调用方已有Top-K Tensor，
只缩短内部scratch生命周期，不改变计算或冻结判定。
CPU lowering通过`csa_indexer_scope_lower/report.json`；先复验B8/131072和B40/131071。
pypto-lib与Simpler工作区仍干净，PyPTO本轮无新增修改。

独立scope复验任务B8 `task_20260922_104541_73573529241`、
B40 `task_20260922_104541_73583214232`均已完成，heap耗尽消失，core完整执行。
两组六个cache/state及Native/PTO未写区域全部PASS；输出/Top-K仍FAIL：
B8分别979/1117个差异，B40分别8435/6148个差异。

B8诊断 `task_20260922_104655_77320613748`确认QR INT8仅1个元素不同（42,851），
FP32 scale最大差`1.44355e-8`；给定相同Native QR和scale时，Indexer RoPE后、Hadamard后
query INT8及FP16 scale均完全一致。误差集中于query27/32/34/37/42。
核对Native fused RMSNorm源码得到三个明确运算顺序：平方和按1024→512→256→128→64折叠，
`1/sqrt(mean+eps)`，以及先`x*rstd*gamma`再求amax；原参考以分段row_sum及
`rstd*max(abs(x*gamma))`替代。当前在本仓库对齐这些FP32边界，同时试验QA与weights_proj
单FP32累加链，避免BF16发布之前split-K重关联。CPU lowering通过
`csa_qr_native_reduce_lower/report.json`；B4/B8/B40三项`qr_precision_*_v1`已排队，尚未验收。

完整CSA的`dsv4_csa_full_replay.py`已补齐并通过ruff/bash静态检查：固定地址图A→B→A，
更新Native长度/positions/页表，使用`DeviceMetadataExecutor`的ExternalEvent与原生复用栅栏；
比较Native/PTO eager及同图replay输出、Top-K、整份allocation和未写区域。
该入口每个variant恢复同一初态，明确不等于连续100步接受轨迹。
首个B4 graph诊断已提交，结果不得提前记为P3通过。

队列等待注意：`task-submit --timeout 50 --wait`超时会自动取消尚未分配设备的任务，
并非只停止观察。本次B8 `task_20260922_105033_92659824506`因此被取消（未执行），
已重新提交同一用例；后续只用`--status`检查等待，不再对pending任务使用限时`--wait`。

## 43. 2026-09-22：完整B4图重放通过，QR与Indexer精度边界

QR单累加链与Native归约顺序版本：B4 `task_20260922_105033_92648422520`完整PASS；
B8重提任务`task_20260922_105920_144872130465`输出PASS（max_abs=0.009765625，
RMSE=0.001658857），Top-K仍76个位置不同，均位于query32；
B40 `task_20260922_105034_92674918555`输出4659、Top-K3476个位置超差。
六个cache/state及Native/PTO保护区均继续通过。

完整CSA图重放首轮`task_20260922_105640_130351314126`在capture前的测试地址签名中失败：
压缩组的合法`None` slot_mapping被调用data_ptr。测试签名保留显式None标记后，
`task_20260922_110453_161540122995`完成PASS：B4/S6，history131071与131073、页表交换，
A→B→A三个变体各30项检查通过；地址固定且使用真实Native ExternalEvent。
graph/eager输出、Top-K及五份完整allocation逐bit一致，Native数值与三套未写区域检查通过。
证据：`full_replay_b4_v2/full_replay.json`。该结果不代表连续100步状态轨迹或完整P3通过。

为区分QA矩阵乘与归一化，诊断捕获Native的`cv_wq_a.matmul`输出，并将生产链归一化
提取成同一`q_proj_qr_normalize`函数供诊断复用，未另写QKV计算。
B8 `task_20260922_110452_161493814420`与B40 `task_20260922_110453_161500220966`
确认：给定Native QA后，B8 QR INT8完全一致但scale有12个末位差；B40仍有8个INT8差异、
55个scale末位差（最大1.74623e-10），说明归一化内部仍有舍入边界不同。
B40实际QA路径比给定Native QA多2个scale差异，最大1.62399e-8，后续仍须核对QA。

同一B40诊断给定Native QR后，Indexer RoPE后14个BF16元素不同。
用已捕获QR/scale与checkpoint INT8权重在CPU计算精确整数点积，非RoPE列比较：
`(acc*activation_scale)*weight_scale`有7个差异，反向依次乘有3个，
`acc*(activation_scale*weight_scale)`为0个。故在本仓库Indexer QB先合并两尺度。
同时使用PyPTO现有`pl.div(..., high_precision=True)`核对Native标量FP32除法，
只调整QR的rstd、量化乘数及其倒数；未修改PyPTO/Simpler/pypto-lib。
这两项改动尚待CPU lowering与B4/B8/B40真机验证，不预记PASS，不变更冻结阈值。

三组`qr_div_scale_*_v1`完成：B4 `task_20260922_111407_215756026152`完整PASS；
B8 `task_20260922_111407_215768123020`输出PASS、Top-K179个位置不同；
B40 `task_20260922_111407_21578172762`输出3832、Top-K2481个差异。
Indexer给定Native QR后的BF16、INT8 query及scale在三组均逐bit相同，证明尺度合并顺序修正有效。
QR与其scale差异未变；生成代码已含`TDIV<DivAlgorithm::HIGH_PRECISION>`，但只读核对
当前PTO-ISA `include/pto/npu/a2a3/TDiv.hpp`发现`TDIV_IMPL`未使用PrecisionType，仍调用vdiv。
因此不能把选项存在当作实际改变精度的证据。下一版在本仓库改用设备端scalar除法计算
rstd与两种scale，复用已有read/write接口，未改PTO-ISA或编译器，未增加host取值。

设备标量除法`qr_scalar_div_*_v1`：B4 `task_20260922_112128_261394329244`完整PASS；
B8 `task_20260922_112128_26140063420`输出PASS、Top-K179个差异（仅query41）；
B40 `task_20260922_112128_261410714842`QR INT8已经完全一致，输出超差降至351个、
Top-K867个差异（query29/41/46/161/184）。六个cache/state及保护区继续通过。
给定Native QA后scale剩B8六个、B40二十一个末位差；B40实际QA路径另有八行scale不同。
为核对sqrt与归约，从现有生产函数提取`q_proj_qr_rms`并复用它采集平方和/rstd。
首次lowering无法推断传入cast临时Tensor的metadata，改为传已有GM Tensor与行偏移后通过，
未修改编译器。B40诊断任务`task_20260922_112531_27738519184`已提交。
Indexer反量化CPU验证证据已保存到`qr_boundary_b40_v1/dequant_order_cpu.json`。

## 44. 2026-09-22：QR平方根的末位定位

`qr_rms_boundary_b40_v1`完成，完整结果仍FAIL。诊断复用生产归约函数采集平方和与rstd；
用同一PTO平方和在CPU按Native顺序执行`1/sqrt(sum/1024+eps)`及真实Tensor/Tensor除法，
240行QR和scale均与Native逐bit一致；用PTO rstd重算则恰好复现21个scale差异。
证据保存在`qr_rms_boundary_b40_v1/rms_cpu_analysis.json`。

新增`qr_boundary`轻量入口，仅给定已捕获Native QA，调用同一生产归一化函数，
可减少定位时整层重复执行。它不是完整CSA或目标权重验收。
首个任务`task_20260922_113032_32463537237`按预期FAIL（QR 0、scale21差异），
同时采出sqrt：PTO VSQRT与CPU sqrt有25个FP32末位差，设备scalar倒数与同输入CPU倒数0差异。
归约及设备标量除法均已排除，剩余scale差异由VSQRT的舍入引起。

为保留只读依赖约束，在本仓库QR中对VSQRT候选值做整数中点比较：
比较输入与相邻FP32平方根中点的平方；24位significand对应的比较整数小于2^52，
全部用设备INT64运算，避免再次引入浮点误差，并保留round-to-even边界。
EPS保证有效输入为正normal值，非有限值保留原结果。CPU以501024个正normal输入及
幂次边界验证、随机给正确sqrt加减1 ULP后恢复，0差异。
CPU lowering通过`csa_qr_sqrt_round_lower/report.json`；轻量B40与完整B8/B40任务记录在
`qr_sqrt_round_v1_manifest.json`，尚待真机结果；冻结阈值和Native算子未修改。

三项完成：轻量B40 `task_20260922_113343_356263113412` PASS，240行QR INT8与scale逐bit一致；
采集的sqrt、rstd与同一平方和的CPU参考均0差异（`sqrt_cpu_check.json`）。
完整B8 `task_20260922_113343_356267121789` PASS：Top-K 24576个位置完全相同，
输出max_abs=0.009765625、RMSE=0.0016587045；全部cache/state和未写区域通过。
完整B40 `task_20260922_113343_356273718818`仍FAIL：输出351、Top-K697个差异，
QR INT8为0差异，scale仅8行不同；给定Native QA后的QR与scale及Indexer边界均逐bit一致。
剩余差异定位到QA累加/发布边界，下一步提取同一QA matmul供诊断复用并采集FP32累加值。

## 45. 2026-09-22：QA累加边界诊断

共享`q_proj_qa`提取初版lowering通过但NPU代码生成失败，任务
`task_20260922_113748_370395527032`未执行PTO core：reshape shape内直接调用tensor.dim
无法生成表达式。恢复为先定义标量再reshape后，任务`task_20260922_113904_37353251679`
完整执行，结果与提取前一致；QA发布BF16有32个值不同，QR INT8相同、scale仍8行不同。
所有提取函数仍位于本仓库，数学运算未另写，未修改依赖。

用同一hidden与BF16权重做CPU FP64点积后舍入BF16：Native有27个、PTO有30个差异。
Native/PTO的32个不同值多处靠近BF16舍入中点或发生相消，不能用更高精度CPU结果
直接替换Native的实际累加行为。CPU结果保存在`qa_boundary_b40_v1/cpu_qa.pt`。

轻量入口增加`--include-projection`，复用生产QA和归一化：
K tile由256临时改为512，任务`task_20260922_114136_382108919779`仍为QA32、QR0、scale8差异；
恢复K256后测试split-K=2，任务`task_20260922_114250_389839224305`为QA34、QR2、scale10差异。
两项试验均未改善，已恢复K256、单累加链。未把调参试验记为正式通过。
下一项观测使用相同Native QA输入，记录linear、连续B、M分块和FP32路径的输出及profiler，
用于确认实际Native算子路径；该观察不修改基线或验收规则。

`task_20260922_114537_5305516886`完成Native QA观测：直接linear与原捕获值0差异，
改为连续B存储有30个BF16差异，按48行分块有32个，FP32输入路径再转BF16有29个。
PTO QA与Native按48行分块结果逐bit完全一致，而与连续B结果有9个差异。
这证明剩余差异与Native大M/转置路径的累加顺序有关，不能靠提高数学精度或改验收基线解决。
初次profiler仅记录aclnnMatmul_MatMulCommon_MatMulV2，继续采集Level1及op args核对实际路径。

完整回归：B4 `task_20260922_114638_9471112876` PASS，输出max_abs=0.0087890625、
RMSE=0.0016623752，cache/state/Top-K/保护区全部通过。
B8完整图`task_20260922_114638_947516243` PASS：history131071/131073/131071，
三个变体各30项检查通过，地址固定、Native外部事件保留，graph/eager整份allocation逐bit一致。
证据为`full_replay_b8_v1/full_replay.json`，仍不替代连续100步状态轨迹。

Level1观测`task_20260922_114822_19671432009`已完成。Native QA实际使用CANN9.0的
legacy `MatMulV2_ND_ND_FP16_FP16_false_true_all.o`，BF16输入、NT布局、22个Cube core，
不是MatMulV3。读取该二进制compileInfo和实际op args，在CPU调用现有`do_op_tiling`：
M48得到tiling_key=98883，M240得到99025，均无跨core split-K。
解码结果保存于`qa_native_observation_b40_v2/cpu_native_tiling.json`。
M48的A L1一次装入全部K4096；M240的A/B L1均按K256分块，L0 K128。
只读核对CANN调度发现大M路径可执行`shift_block_access`，循环移动K访问起点；
当前继续核对该累加顺序，不把tiling差异本身视为已修复，也不修改Native基线。
CPU tiling查询首次因缺少`ori_shape`失败，补齐字段后查询成功；首次保存bytes到JSON失败，
改为保存hex、uint32及字段解码后成功。上述失败均未运行NPU计算。

QA K顺序观测复用`diagnose_qa`/生产`q_proj_qa`，仅把两输入的K维同步循环排列，
不写另一套matmul。`task_20260922_120239_140435110970`枚举K256起点，
`task_20260922_120436_148213828070`按Native L0粒度枚举K128起点，均完成exit=0。
该exit=0只表示观测完成及给定Native QA的归一化仍PASS，不表示观测QA已通过。
两份`qa_k_rotations.json`显示，N96分块中的第1/5/7块没有任何统一起点能完全对齐；
K256观测另有第3块不能对齐。故单一循环起点假设不足，不能据此写入生产QA。
生产QA仍为K256、单累加链；结果与完成时源码hash见`qa_k_rotation_manifest.json`。
补充CPU六档形状tiling查询在`qa_native_observation_b40_v2/cpu_six_shape_tiling.json`。
本轮未改变数值容差、Native基线或依赖仓库，B40完整结果仍为输出351/Top-K697个差异。

## 46. 2026-09-22：定位Native QA的正反向K调度

上轮取得实际tiling及K起点观测，属于有效进展；本轮继续核对生成代码。
先按Native L0 K128测试生产QA，任务`task_20260922_121101_198845131022`完成FAIL：
QA32、QR0、scale8差异，K128起点观测也与K256版本相同。已逐字节恢复试验前K256源码，
未把无效调参保留为修复。

CPU尝试调用CANN动态入口缺少range、直接静态入口缺少compute context，均在编译前失败。
随后复用安装版`_binary_constant_branch`成功生成同形状CANN MatMulV2代码；
不修改CANN/TBE或任一依赖。首个证据为
`qa_native_codegen_b40_v3/kernel_meta/csa_native_qa_observe.cce`。
可复跑的六档CPU观察脚本及生成源位于`qa_native_codegen_six/`，六项生成均完成。

M96/144/192/240的生成代码给出相同规则：令`g=column//96`、`k=0..15`，
第0～8个N组的K256块顺序为
`(3*(g//2) + (k if g%2==0 else 15-k)) % 16`；第9/10组处于重叠尾区，不移动K顺序。
奇数组不仅移动起点，还反向遍历256块，但块内元素仍保持正序，解释了前轮单纯rotation
无法完全匹配第1/3/5/7组。M24/M48生成代码没有这项调度。

本仓库`q_proj_qa`按上述规则调整现有K遍历；N tile改为32，避免跨Native的96列顺序边界。
保留K256、单FP32累加器、已有QA BF16发布及QR函数链；未改Native基线。
首版仅M240，轻量任务`task_20260922_121633_20855309404`完成PASS：
QA BF16、QR INT8和scale全部0差异。随后按已生成源扩展至M96/144/192，
六档完整CSA对照任务与源码hash记于`qa_native_full_v1_manifest.json`，尚待结果。

六项完整对照均已完成PASS：B4 `task_20260922_121803_213381718645`、
B8 `task_20260922_121803_21338577904`、B16 `task_20260922_121803_213389714639`、
B24 `task_20260922_121803_213399211069`、B32 `task_20260922_121803_213410912351`、
B40 `task_20260922_121803_213426623243`。
B40 history131071，其余history131072；各项输出、Top-K、六个cache/state及全部未写区域通过，
B40原先351个输出超差与697个Top-K位置差异均归零。
以同一源码补齐其余12组长度组合，完整18组清单在`reference_matrix_v2/manifest.json`；
同时提交B40 A→B→A完整图对照，尚待结果。上述均为参考权重验证，正式目标checkpoint未加载。

## 47. 2026-09-22：18组矩阵的叶内并列排序与连续轨迹入口

`reference_matrix_v2`全部完成：16/18通过，仅B32/B40的history131073失败；
18组最终输出、cache/state和未写区域全部通过。失败都为query144的Top-K第277/278位互换：
Native为28258/26524，PTO为26524/28258；PTO两分数均为FP32位型979142915
（`0.0008412750321440399`）。B40图任务`task_20260922_122043_223231432503`
A→B→A仅中间history131073的Native Top-K对照失败，graph/eager输出、Top-K及完整allocation
仍全部逐bit一致，两个A变体通过；不记为完整图数值验收通过。

只读核对Native arch22：`S2_BASE_SIZE=2048`，每块先`SortAll`，随后`MergeSort`
把新块放在累计结果之前。两个不同索引分别属于2048块12/13，却在同一个PTO 4096半叶内。
因此原来只调整叶间合并不足；本轮让半叶内的两个2048块也按后块优先合并，
单叶4096/8192路径采用相同块优先顺序，块内Sort32/归并和所有score不变。
复验B32/B40历史131073及B4连续100步，任务与源码hash在`topk_chunk_v1_manifest.json`。

新增`dsv4_csa_continuous.py`及`continuous`队列入口，复用现有完整CSA、Native fixture、
接受长度修正与metadata executor。先完成整段eager，再从同一初态执行同样graph轨迹；
每步保留cache/state，核对输出、精确Top-K、六个cache/state和相对上一状态的未写区域；
整份物理allocation/output/Top-K的SHA256核对graph与eager逐字节一致。
输入是独立确定的测试数据；仅断言在核执行后取回device结果，不以回读值驱动计算。
末次graph replay采集NPU profiler，不以Python调用次数替代设备执行证据。

短测`task_20260922_122606_247459026237`完成PASS：B4、mixed平均推进3.8，
eager5步和graph5步均通过；`hundred_steps_completed=false`，不能计作G01完成。
`continuous_b4_smoke_v1/graph_profile`记录33条device kernel，其中包含
`aicore_kernel_mode_0_mix_aic`及`simpler_aicpu_kernel_exec_e483426d38182c24`。
该入口不覆盖请求换位/退出、prefix共享、padding等其他P3用例；P3整体仍待验。

## 48. 2026-09-22：参考18组全部通过，连续轨迹第6步诊断

上一轮完成了QA/Top-K修复与连续测试入口，属于有效进展。本轮先核对队列终态：
`task_20260922_122946_26248737053`与`task_20260922_122946_2624943235`均exit0，
B32/B40 history131073完整对照通过，原并列候选交换已消除。
随后按同一生产源码hash重跑其余16组；`reference_matrix_v3/manifest.json`保存18项
任务、命令和源码hash，`summary.json`保存队列终态与报告路径。18项全部exit0/PASS；
逐项核对输出、精确Top-K、六个cache/state以及Native/PTO未写区域均通过。
原冻结容差未变，输出max_abs范围0.0078125～0.01171875（按atol+rtol判断），
RMSE范围0.0016167450～0.0017273125。这里只验48分片cann_recipe参考权重。

B40完整图复验`task_20260922_123814_292352911375`exit0/PASS，结果在
`full_replay_b40_v2/full_replay.json`。A→B→A的三个变体均通过Native数值/状态/Top-K，
同地址graph/eager输出、Top-K及完整物理allocation逐bit一致，保留Native external event。
这项不等于连续状态或服务forward已验收。

B4 mixed100步`task_20260922_122946_26250177984`exit1：eager step0～4通过，
step5起点均为131090时，仅Indexer INT8 cache的4个值不同，均相差1；
位置为page2069/offset5/channel8、34、109、111。输出、Top-K、其他cache/state和未写区域通过；
graph尚未执行，不能将五步短测外推为100步通过。

新增`dsv4_csa_compressor_boundary.py`及continuous的`--diagnostic-step`，直接复用生产
project/pool/write函数，分别从Native/PTO前一步state重算并采集量化前边界；CPU lowering通过。
首诊断`task_20260922_124304_323026314904`复现第6步失败，但诊断调用漏了full CSA
调用方的RoPE half-split→interleaved转换，导致诊断与生产cache不一致；该诊断数值无效。
已在测试入口补齐相同系数布局转换并增加投影/state边界采集，生产core未改变，待复验。

同步更新计划、README与本日志顶部状态，旧机器CANN9.2/两卡内容明确标作历史。
正式目标权重仍等待用户通知；未轮询下载，未修改PyPTO、Simpler或pypto-lib。

## 49. 2026-09-22：Indexer compressor的K顺序与第12步Top-K边界

修正诊断RoPE布局后，`task_20260922_124727_33722137661`完成有效边界观测。
诊断cache与实际生产cache完全一致；给定Native或PTO前一步state仍有相同的归一化/RoPE差异。
捕获的FP32 value投影有4893个不同值，score加APE后有4858个；归一化有1个BF16差异，
RoPE后有39个BF16差异。故不能把4个INT8 cache差异仅归因于跨步state输入。

只读核对当前安装的Native compressor源码，并确认与本仓库源文件哈希相同：
arch32的uniform S6、head128、小T路径使用8个N组，每组16列，K_L1_BASE=256；
各N组从不同的K256块起点循环访问。此前PTO使用K512且所有列使用相同起点。
本仓库`decode_indexer_compressor.py`改为K256/N16，按每半head内的16列组移动K起点；
两条投影均保留单FP32累加链，两个overlap半区重复同一顺序。证据及源码hash在
`indexer_projection_order_v1/`；这项限定于当前冻结的TP1/S6形状，未修改Native或依赖。

`task_20260922_125112_351974429258`在原失败step5的FP32 value/score投影、归一化、
RoPE、cache及进入该步的Indexer state均完全一致，4个INT8差异已消除。
同一任务继续执行mixed100步，在eager step11（第12步）因query20的14个Top-K位置不同而停止；
step0～10全部通过，step11输出、六个cache/state和所有未写区域通过，graph尚未执行。
另一个任务`task_20260922_125112_351978832631`完成B40/history131073完整对照PASS。
第48节18组矩阵在本次生产代码调整之前；当前版本其余17组仍需复验，不能直接继承旧PASS。

新增`dsv4_csa_query_boundary.py`和`--query-diagnostic-step`，复用生产QA、QR及Indexer函数，
捕获Native边界、完整Indexer cache、weights、query及Top-K供后续比对。
`task_20260922_125457_360974211228`复现step11的14个Top-K差异，且QA、QR、QR scale、
Hadamard前BF16、INT8 query与query scale均逐bit相同。证据在`continuous_b4_query_v1/`。
剩余工作核对head weights、QLI分数与并列排序；本记录不提前归因，也不记为P3通过。

## 50. 2026-09-22：A3高精度TDIV归因、Native对比与独立复现

按用户要求新增独立说明
[PTO_ISA_A3_TDIV_HIGH_PRECISION_REPRO.md](PTO_ISA_A3_TDIV_HIGH_PRECISION_REPRO.md)，
并提供`repro_tdiv_high_precision.py`及本机队列包装`run_tdiv_high_precision_repro.sh`。
复现不依赖DSV4权重、vLLM或Native custom op，只比较相同输入的默认TDIV、HIGH_PRECISION
TDIV和设备标量除法。输入16×256，种子20260922，均为正的normal FP32；
参考候选以FP64计算，并对4096个值逐一用精确有理数核对最近偶数舍入。

`task_20260922_125910_368689920621`在A3设备0完成exit0，状态`REPRODUCED`：
默认与高精度选项0差异；两者相对正确舍入FP32均有244个1 ULP差异；设备标量路径0差异。
exit0表示观察完成，不能解释成高精度验收通过。完整输入/输出、IR/C++、源码摘录及报告位于
`tdiv_high_precision_repro_v1/`。独立PTOAS编译同一IR也保留HIGH_PRECISION参数。
首次误用level2在显式tile地址处失败，改用level3后exit0；这项调用错误与TDIV精度无关。

补充只读核对后，需要收紧第43节的组件定性：当前PTO-ISA `docs/isa/TDIV.md`明确声明
高精度选项仅适用于A5、A3会忽略它。PyPTO已把属性传入IR，PTOAS0.61已生成
`TDIV<pto::DivAlgorithm::HIGH_PRECISION>`，公共ISA入口也继续传参；A2/A3实现没有精度分支，
两种模式均调用同一vdiv。因此这是PTO-ISA已声明的A3能力缺口，不是本例证明的前端/编译器
丢参数问题，也不直接定性为违反现有ISA文档的实现bug。PyPTO/PTOAS可补平台能力提示。

Native QR在同一A3上用设备端标量C++除法计算每行rstd、量化乘数及scale，再执行向量乘法；
GetValue/SetValue位于aicore内，并有V_S/S_V事件依赖，没有CPU回读计算。
因此不能把差异解释成Native用了另一种硬件，或A3整体无法达到这种精度。
当前材料只能证明A3高精度TDIV未实现且已文档化；没有证据说明其原因是性能、排期或硬件绝对限制。
A5源码有独立补偿分支，但本次未在A5、A2或FP16上做数值测试，亦未对两路径反汇编或测性能。

本次归因、脚本与说明均写在vllm-ascend-dsv4-pto；未修改PyPTO、Simpler、PTO-ISA或pypto-lib。

## 51. 2026-09-22：TDIV说明与独立复现单独提交并推送

按用户要求，仅提交TDIV说明、两份独立复现脚本、结果摘要、源码链/PTOAS证据、
任务manifest及包含原始输入输出/生成文件的证据包，共9个文件。证据包只保留相关
QR日志第43～46节作为上下文，不夹带其他CSA工作区修改。
提交为`9031197a82534fa30f073415dd6657620edc0f53`：`test(pto): document and reproduce A3 TDIV precision limitation`；
commit正文详细记录组件归因、Native标量路径、4096样本实验、版本及验证范围。
以当前分支上一条提交的nalinaly身份签署，仅在该次命令中提供Git身份。

用户明确要求不进行提交检查，因此未运行提交检查，也未重跑已完成的真机实验。
此前隔离lint环境的工具下载失败，未执行检查；未再重试，未改动主运行环境。
`git push origin HEAD:refs/heads/dsv4-flash-pto`成功，远端从e771542更新到9031197。
本日志和其他CSA接入改动继续保留在工作区，未包含在该独立提交中。

## 52. 2026-09-22：连续第12步的head weights与QLI系数定位

上一轮已按用户要求完成TDIV独立提交并推送，属于已完成的进展；本轮回到P3剩余项。
先检查保存的query20候选：14个位置差异是7对相邻候选交换，PTO对应分数并不相等，
不能按并列排序问题直接处理。首次CPU读取漏去Native Top-K的中间单维，产生广播后
报错；按PTO shape显式reshape后得到有效的14个位置比较，未修改测试或生产的数值断言。

用捕获的相同INT8 query/key做精确INT32点积，按Native的FP16 head系数及QK边界
在CPU计算候选分数：512个候选排序与Native完全一致，PTO分数有429个值不同，
最大7.7649e-8。CPU FP64 head归约只用于定位，不作为Native Cube累加的替代基线。
进一步按各head拟合分数差异，head21的系数从-2.5033950805664062e-6改为
-2.4437904357910156e-6（相差一个FP16 subnormal ULP）即可使PTO分数与CPU预测
最大只差2个FP32 ULP；其余head拟合项约1e-11。此时尚为数值线索，不提前认定生产系数不同。
证据在`continuous_b4_query_v1/score_cpu.py`、`score_cpu.json`和`score_coefficient_fit.json`。

将已有生产head权重投影及QLI系数计算提取为`indexer_weights_project`、
`indexer_head_coefficients`，完整CSA与诊断调用同一函数，运算、dtype发布和依赖顺序不变。
诊断补采Native原始/缩放后weights、PTO weights、FP16 coefficients及投影权重；
CPU lowering通过，源码快照与manifest在`weights_coefficient_boundary_v1/`。
真机任务`task_20260922_133302_80585723723`已提交，B4/12步、诊断step11，等待结果。

`task_20260922_133302_80585723723`完成：原step11仍有14个Top-K差异；
新采集确认仅weights[20,21]有一个BF16间隔（7.6293945e-6），其FP16系数相差
5.9604645e-8，与CPU拟合一致；QA/QR/query边界仍全部逐bit相同。

新增`dsv4_csa_weights_boundary.py`，用捕获输入重放同一生产weights/coefficients函数，
并采集Native linear的Level1 profiler。`task_20260922_133815_134328529140`完成，
Native linear与捕获模块输出逐bit相同，六档形状全部实际调用
`aclnnMatmul_MatMulCommon_MatMulV2`，ND/BF16、NT输入，记录了具体binary及op args。
重复该组隐藏值至M24/48/96/144/192/240时，旧PTO分别有1/2/3/3/6/7个weights差异，
均传递成对应系数差异。没有把此诊断FAIL记为完整CSA或其他输入的结论。

用当前CANN已安装的MatMulV2生成器取得六档代码，看到下列实际K顺序。记r=row//16、
g=head//16，k为当前顺序中的块序号；块内保持正向，FP32累加器不拆分：

| M | K块大小 | 块访问顺序 |
| --- | --- | --- |
| 24 | 512 | `(4*(g//2) + (k if g%2==0 else 7-k)) % 8` |
| 48 | 256 | `(8*(g//2) + (k if g%2==0 else 15-k)) % 16` |
| 96 | 256 | `(5*(r//2) + (k if r%2==0 else 15-k)) % 16` |
| 144 | 256 | `(4*((r%8)//2) + (k if r%2==0 else 15-k)) % 16` |
| 192 | 256 | `(2*(r//2) + (k if r%2==0 else 15-k)) % 16` |
| 240 | 256 | `(2*((r%14)//2) + (k if r%2==0 else 15-k)) % 16` |

本仓库weights投影改用N16/K256 tile，M24把512块分成两个保持内部顺序的256块；
依照上表选择K访问次序，保留原有BF16发布、BF16缩放及FP16权重输入边界。
CPU lowering通过；源码证据映射与任务/hash在`weights_korder_v1/`。
`task_20260922_134414_191079626641`完成PASS：六档weights和FP16系数全部0差异。
未修改PyPTO、Simpler、PTO-ISA、pypto-lib或Native基线。

## 53. 2026-09-22：连续轨迹推进到第25步，定位单个输出超差

新K顺序下的B4 mixed100步任务`task_20260922_134414_1910889702`完成exit1：
eager step0～23全部通过，第12步Top-K差异已消除；step24（第25步）仅一个输出
[18,1531]超出冻结容差，Native=-0.05322265625，PTO=-0.06396484375，
差值0.0107421875，允许值0.0105322264。该步Top-K、六个cache/state和未写区域均通过；
graph尚未执行，完整100步/P3仍未通过。

新增`OutputBoundary`在指定失败步捕获Native投影输入、WO-A结果及可观察到的WO-B
量化边界，再把Native投影输入按既有group布局送入同一生产`decode_o_proj_tp1`。
该诊断只用于区分输出投影与上游attention误差，不改变生产计算或验收容差。

首个输出诊断`task_20260922_135558_276514223294`使用`--steps 25`，eager/graph各25步
通过。但fixture容量按`history + 6 * steps + 12`生成，25步对应131233，原100步对应131683；
这会改变物理页映射及随机初始化shape。两个任务从step0起输出、Top-K和历史cache指纹已不同，
故该PASS不能作为原失败已消除的证据。对比见`output_boundary_step24_v1/fixture_capacity_comparison.json`。

按原100步配置重跑`task_20260922_140452_30654656981`，准确复现step24同一元素超差。
全部25步PTO指纹与原任务相同，保存的hidden、positions、Native/PTO输出、Top-K和分数均逐bit相同；
诊断hook没有改变这次失败。证据见`continuous_matrix_v2/b4/reproduction_comparison.json`。
给定Native投影输入后，PTO输出投影的max_abs=0.00390625、RMSE=0.0001557545183，冻结阈值通过；
原失败元素变为-0.053466796875，与Native差0.000244140625。精确比较仍有3094个不同值，
不宣称输出投影逐bit一致；主要误差进一步定位到投影之前的Query/attention链。

## 54. 2026-09-22：最新18组单层回归与六档连续轨迹

`reference_matrix_v4/manifest.json`记录当前完整生产源与测试源hash及18个任务，全部已终态exit0。
B4/8/16/24/32/40 × history131071/131072/131073各18项检查全部通过，共324项；
输出max_abs最大0.01171875，RMSE最大0.001727312454，均满足原冻结混合容差，Top-K精确一致。
每项六个cache/state及Native/PTO未写区域通过；汇总见`reference_matrix_v4/checks_summary.json`。
本版本已包含第49节Indexer compressor和第52节head weights累加顺序修正。
这是参考单层结果，不替代正式ModelSlim权重、连续100步或服务派发验收。

另提交六档mixed100步，固定history131071/seed1024/容量131683；B4仅在step24启用输出诊断。
`continuous_matrix_v2/manifest.json`保留命令、源hash与任务ID，结果如下：

| B | 任务 | 首个失败step（从0计） | 输出超差元素数 |
| --- | --- | --- | --- |
| 4 | task_20260922_140452_30654656981 | 24 | 1 |
| 8 | task_20260922_140452_306550511649 | 6 | 1 |
| 16 | task_20260922_140452_306555913574 | 6 | 1 |
| 24 | task_20260922_140452_30656364507 | 3 | 1 |
| 32 | task_20260922_140452_30657217268 | 0 | 5 |
| 40 | task_20260922_140452_30658344884 | 0 | 5 |

六个任务均因输出断言exit1，失败步Top-K、全部cache/state和保护检查通过，均未进入graph阶段。
B8/B16的失败位置同为[37,450]；B32/B40均在query191的5列。后续用同一生产Query与稀疏attention
函数补采边界，继续定位；未修改阈值或将观察性诊断计为P3通过。

## 55. 2026-09-22：稀疏attention的softmax分块与BF16概率边界

扩展输出诊断，复用同一生产qkv_proj_rope、sparse_attn_csa_tp1及decode_o_proj_tp1；
分别给定PTO Query/PTO cache、Native Query/PTO cache、Native Query/Native cache。
两份诊断kernel CPU lowering通过；B4任务`task_20260922_141320_330799313324`、
B32任务`task_20260922_141321_33080982072`均准确复现原失败，完整PTO逐步指纹未变。
拆段重算输出与实际完整PTO输出逐bit相同，故诊断边界有效。

B4有7个、B32有28个主Query BF16值不同，QR INT8和scale全部相同；但原失败行
B4 query18、B32 query191的Query均完全一致。即使同时提供Native Query和Native cache，
仍保留原1个/5个输出超差，因此当前失败进一步定位到稀疏attention内部；不把它误归因于Query。
保存的Native/PTO selected KV只含实际选中的640行，可用于CPU分析；证据在`attention_boundary_v1/`。

只读核对实际A3入口`npu_sparse_attn_sharedkv`及安装包arch32源码，vector源文件hash与本仓库相同。
Native s2BaseSize固定512，先处理原始窗口，再处理512个压缩候选；SoftmaxFlashV2以sink为初始
max、1为初始sum，使用累计max后再将概率发布为BF16。此前PTO把640个候选分为5个128块，
各块按局部max转换BF16后才合并FP32 PV。这两种数学等价分解的BF16舍入不同。

用捕获的相同Native Query/cache在CPU以FP64点积、FP32 softmax及BF16概率边界作归因，
只比较不受inverse RoPE影响的前448列，不能替代Cube归约或真机验收：
B4失败行的Native/PTO差异11792个，模拟原128块为11788个，累计max的128块为3951个，
累计max的512块为125个；B32失败行对应11540、11539、4247、3个。
源码和完整CPU脚本/结果见`native_softmax_source.json`、`softmax_order_cpu.py/json`。
据此在本仓库调整512候选softmax、sink累积及概率舍入边界，Cube仍以128候选分片搬运与计算
以满足A3片上容量；保存改动前完整源，后续以原100步配置真机验证，不预记PASS。

512候选实现初次lowering发现测试改写中使用了Tensor构造替代显式Tile构造，修正为pl.tile.full；
动态Tile下标改用已有pl.tile.slice。随后H16临时区229888字节、H8版本213440字节，
均超过当前运行时188416字节可用限制；调整PV左右半区的读取/计算次序缩短临时值存活，
H8版本`softmax512_lowering_v5.json`通过。以上均在本仓库修改，没有改动编译器或运行时限额。
原四次失败及最终lowering分别保存在`attention_boundary_v1/softmax512_lowering*.json/log`。
`softmax512_v1/manifest.json`记录新生产源hash和B4/B32原100步真机任务，提交时不预记PASS。

首轮真机`softmax512_v1`的B4/B32任务分别为`task_20260922_143047_365757831347`、
`task_20260922_143047_365763111303`，均在PTOAS编译阶段exit1，未执行数值比较：
单列pl.tile.full生成了不满足32字节行对齐的row-major Tile。初始化改为从已加载列向量
继承布局，CPU signature-only完整编译`softmax512_compile_v6.json`通过。

`softmax512_v2`的B4任务`task_20260922_143536_376279929252`、B32任务
`task_20260922_143536_376286318234`完成数值执行，但均在step0输出失败，cache/state、
Top-K及保护区继续通过。B32拆段诊断显示每个AIV前8个head已基本与Native对齐，
后续head组误差很大；生成C++中sm_old_m/sm_old_l的TASSIGN地址不随动态sm_part变化。
完整生成文件、源码hash和观察见`dynamic_slice_generated_aiv.cpp`、
`dynamic_slice_generated_source.json`；尚未单独缩减组件复现，不能把组件归因预记为已证明。

尝试静态展开head组时，当前前端在ConvertToSSA报m_iter/l_iter定义域错误，保留失败源及报告。
本仓库改为使用已有GM统计buffer按显式head行偏移读取max/sum，避免动态Vec Tile切片；
不增加host取值，也不修改依赖仓库。`softmax512_compile_v8.json`已完成CPU编译PASS，
`softmax512_v3/manifest.json`记录最新B4/B32原100步验证任务及源hash，结果待归档。

## 56. 2026-09-22：softmax修正后B4完整100步通过，Indexer归一化边界

`softmax512_v3/b4`任务`task_20260922_144340_398476012782`已终态exit0。
保留原history131071、seed1024、mixed100及fixture容量131683，eager/graph各100步通过，
共3700项检查；输出、精确Top-K、六个cache/state及两侧未写区域均满足冻结门槛。
同轨迹graph/eager的输出、Top-K及五份完整allocation指纹逐bit一致，地址固定，
使用真实Native ExternalEvent与接受长度修正，步间保留cache/state。输出max_abs最大
0.01171875、RMSE最大0.0008246047073，均通过混合容差；原step24输出超差已消除。
该结果只覆盖B4的mixed100，不将P3全部用例记为通过。

`softmax512_regression_v1/reference_matrix/`的18组任务全部终态exit0，
B4/8/16/24/32/40 × history131071/131072/131073共324项通过，
输出max_abs最大0.0078125、RMSE最大0.0003858533164，Top-K逐元素一致。
核对全部生产及测试源hash与提交任务时manifest一致；汇总见`reference_matrix/checks_summary.json`。
这是参考单层回归，不替代正式ModelSlim权重验收。

其他五档连续测试结果如下，均只有Indexer INT8 cache失败；截至失败步的输出、Top-K、
其余cache/state和保护检查全部通过，均未进入graph。详见`continuous_summary.json`及manifest。

| B | 任务 | 首个失败step（从0计） | INT8差异数 |
| --- | --- | --- | --- |
| 8 | task_20260922_144645_4072666694 | 56 | 5 |
| 16 | task_20260922_144645_407274911903 | 56 | 5 |
| 24 | task_20260922_144645_407283625469 | 10 | 3 |
| 32 | task_20260922_144645_407261924425 | 10 | 3 |
| 40 | task_20260922_144645_40729385948 | 10 | 3 |

B32无诊断任务`task_20260922_144340_398486827423`也在step10出现同样3个INT8差异。
复用生产compressor的诊断给定Native/PTO各自前态，两个前态逐bit一致，value投影和
score+APE也与Native逐bit一致。首个可观察差异为request18/token110/position131111的
归一化输出列31：Native=-1.1015625，PTO=-1.109375；Hadamard后29个BF16值不同，
最终物理slot[18565,9]的列77/80/83相差1。两次拆段重算cache均与实际PTO完整allocation
逐bit一致，诊断没有另写投影或归一化。证据为`b32_compressor/compressor_boundary.json/pt`。
上述证据将问题缩小到池化/归一化，不归因于投影、前态或softmax修正。

只读核对Native arch32 `compressor/rms_norm.h`及`compressor_vector_comm.h`：
128维平方先按列合并两个64段，再WholeReduceSum；对sqrt结果直接做向量Div后乘gamma。
当前PTO分别归约两个64段再相加，并以recip(sqrt)乘输入，运算边界不同。
后续在本仓库对齐该顺序并用原100步配置验证；不改依赖仓库、Native基线或冻结阈值。

B4 graph step99的profiler记录9次Simpler AICPU任务及9次对应AICore kernel-mode执行，
并含Native compressor/QLI/SAS metadata任务；详见`softmax512_v3/b4/graph_profile_summary.json`。
这是设备执行证据，不以Python调用计数替代，也不计作P5性能验收。

## 57. 2026-09-22：对齐Indexer RMSNorm归约和除法顺序

仅修改本仓库`decode_indexer_compressor.py`的RMSNorm：两个64列平方向量先逐列相加，
再作row_sum；保持A3向量sqrt，对输入做row_expand_div后乘gamma。
去掉原先的分段归约相加及recip乘法；投影、池化、状态写回、RoPE和量化未改。
修改前完整源与Native源码hash保存在`indexer_rms_order_v1/`。
复用生产compressor的CPU signature-only完整编译（含PTOAS）通过，见`compile.json`。

`indexer_rms_order_v1/manifest.json`保存全部源hash和100步任务：
B32 mixed/step10诊断`task_20260922_150007_157947930493`，
B8 mixed/step56诊断`task_20260922_150007_15802822670`；
另扩展G02边界轨迹，B4 all1/all6/reject_then_accept分别为
`task_20260922_150007_15811517842`、`task_20260922_150007_158180514516`、
`task_20260922_150007_15826107853`。均保留100步容量，结果待归档；
第56节的已通过结果属于RMS顺序修改前版本，不提前外推新版本通过。

上述五个任务均已终态。B4 all1/all6/reject_then_accept三项各eager100+graph100通过，
每项3700检查；这补充G02的B4参考轨迹证据，不扩大到其他BS或正式目标权重。
B32 step10诊断的两种前态下normalized、Hadamard、完整INT8 cache以及value/score投影
全部逐bit相同，原step10差异消除；连续执行在step21的slot[20627,19]列61/62/117
出现3个INT8差异，其余检查通过。B8仍在step56有5个INT8差异：诊断前态和投影逐bit相同，
request7/token47/position131287的归一化列35为Native=-1.234375、PTO=-1.2265625，
Hadamard后32个值不同。汇总见`indexer_rms_order_v1/summary.json`。

## 58. 2026-09-22：Indexer八行池化的概率归一化与归约顺序

只读核对Native arch32 `compressor_block_vec_perf.h`、`soft_max.h`及`compressor_vector_comm.h`：
ratio4 overlap按前/当前压缩组交错排列8行；ColumnSoftMax先全8行max/exp，再按8→4→2→1
树形求和并逐元素除以sum；KvMulReduceScore先乘归一化概率，再用同一树形归约。
本仓库原参考路径按末token起始，依次做online max/PV累计，最后除以累计sum，
数学等价但FP32运算顺序及除法位置不同。这是源码差异，仍须用真机验证其数值影响。

在本仓库保留投影、状态选择和写回、RMSNorm/RoPE与量化链，只对齐上述八行池化顺序。
每个pool worker使用独立、有界的GM窗口暂存；仍按既有state ring与本次token覆盖规则取值。
生产函数原有pooled_kv缓冲改由调用方提供，诊断复用相同函数保存池化输出，未另写投影或池化。
修改前源与Native源码hash保存在`indexer_pool_order_v1/`；编译及NPU结果待归档。

生产compressor诊断及完整CSA CPU编译均已通过（含PTOAS），分别见`compile.json`和
`full_compile.json`。六档mixed100及B4 all1/all6/reject_then_accept共9个NPU任务
已提交，命令、源hash和任务ID见`manifest.json`，任务前缀`task_20260922_152351_`。
提交后仍为pending：共享队列中另一个16卡任务正在执行；维护模式关闭，auto候选为0～15。
遵守队列分配，不直接占卡或干预其他任务；未将排队任务预记为通过。

用捕获的投影和前态在CPU重建8行窗口时，online与Native树形两种版本在B8 step56、
B32 step10都能得到与Native相同的BF16归一化结果；CPU exp/div/sqrt不能充当A3指令
舍入oracle，因此这项CPU计算无法区分设备边界原因，也不证明新池化版本已修复。
输入hash和观察见`cpu_attribution_limit.json`；最终仍以排队中的真机复验为准。

## 59. 2026-09-22：八行池化失败点复验与G08生产流延迟

上一轮为实质进展：对齐RMS及八行池化、完成CPU完整编译并提交真机任务；本轮首先从队列
核实原9个任务已获得设备并运行，没有重提。源hash与提交时一致。
B32/B40 step21、B8/B16 step56及B24 step10诊断的normalized、Hadamard、完整cache、
value投影和score+APE全部逐bit一致；截至观测时已消除这些旧失败点，100步任务仍在执行，
不能提前记为完整通过。另提交最新18组参考单层回归，见`reference_matrix_v5/manifest.json`。

新增测试驱动`dsv4_csa_cross_stream.py`，复用现有`dsv4_csa_full_replay.main`的完整
Native/eager/graph A→B→A比较；没有复制CSA计算或更改生产executor。
测试进程内临时包装`DeviceMetadataExecutor.submit`，只对使用BatchDescriptor/ExternalEvent
的PTO调用，在选定Native生产stage的原task.run之前加入设备`torch_npu.npu._sleep`；
保留原stage/group、ready事件和buffer复用fence。Native参考计算不加入延迟。
三种variant分别使用0、1000000、10000000 cycles，eager/graph对应相同延迟；
设备timing event记录实际生产耗时，取值仅在共享测试的数值断言同步之后。
测试断言生产/消费stream不同、使用外部事件、普通和延迟graph均执行，且最大延迟的
生产耗时中位数大于零延迟；不添加host sleep或额外全局同步。

G08初轮为B4/B40 × COMPRESSOR/INDEXER/ATTENTION，共6个任务，
任务前缀`task_20260922_153744_`/`task_20260922_153745_`，完整命令及源hash见
`full_cross_stream_v1/manifest.json`。这是固定BS下三种生产stage的跨流检验，
不替代G04～G07请求变化、padding与prefix共享，也不预记PASS。

## 60. 2026-09-22：八行池化版本的参考矩阵与连续轨迹终态

`reference_matrix_v5`的18组任务均终态exit0，B4/8/16/24/32/40 ×
history131071/131072/131073共324项检查通过。全部源hash与任务manifest一致；
输出max_abs最大0.0078125、RMSE最大0.0003858533164，Top-K精确相同，
六个cache/state及两侧保护区通过。证据为`reference_matrix_v5/checks_summary.json`。
这是包含Indexer RMS和八行池化修正的当前参考版本，不替代正式ModelSlim权重验收。

`indexer_pool_order_v1`的9个任务均已终态，汇总见`summary.json`。
保留history131071、seed1024、100步及容量131683，不通过缩短轨迹更换随机缓存。
B4 mixed、B8 mixed、B4 all1/all6/reject_then_accept全部各eager100+graph100 PASS，
每项3700检查；graph/eager输出、Top-K与五份完整allocation指纹逐bit相同。
对应任务后缀分别为`299931815138`、`29992705222`、`299957821936`、
`29996191717`、`299966810787`，完整前缀均为`task_20260922_152351_`。

其余四档仅输出失败，失败步的Top-K、六个cache/state及全部保护检查通过；
均未进入graph阶段。以下step从0计，容差仍为atol/rtol=0.01/0.01：

| B | 任务后缀（同上前缀） | 首个失败step | 输出位置 | Native | PTO | 绝对差 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 29994091674 | 63 | [31,3469] | -0.028564453125 | -0.0181884765625 | 0.0103759765625 |
| 24 | 299949726524 | 63 | [31,3469] | -0.0289306640625 | -0.0181884765625 | 0.0107421875 |
| 32 | 299918423829 | 63 | [31,3469] | -0.0289306640625 | -0.0181884765625 | 0.0107421875 |
| 40 | 299953826218 | 46 | [185,3015] | 0.0174560546875 | 0.006988525390625 | 0.010467529296875 |

旧Indexer失败点的两种前态诊断均已逐bit对齐：B32/B40 step21、B8/B16 step56、
B24 step10的normalized、Hadamard、完整INT8 cache和value/score投影均无差异。
不能把主compressor FP32 state的容差通过描述为逐bit相同，也不把新输出失败归为Indexer cache失败。

## 61. 2026-09-22：连续轨迹后期输出边界复现

`continuous_output_late_v1`的B16任务`task_20260922_154305_133225610263`与B40任务
`task_20260922_154305_13324142666`，分别在step63和step46复现同一输出失败。
仍使用100步fixture，复用生产Query、稀疏attention和O projection函数采集边界。
拆段PTO Query/PTO cache重算与完整PTO输出逐bit相同，诊断未替换计算链。

B16失败query31、B40失败query185的Query均与Native逐bit相同，QR INT8和scale也一致。
给定Native O projection输入后，两档输出均满足冻结门槛；该边界仍有BF16末位差，
不能表述为O projection逐bit一致。

B16同时给定Native Query和Native cache后，输出仍有2个超差位置[31,2804]和[31,3469]，
max_abs=0.01171875。失败query31的attention projection输入有101个BF16差异，
其中head51占100个、head49占1个，含92个NoPE和9个RoPE列，最大差0.000244140625。
故B16仍需定位稀疏attention内部，不能只修主compressor后便认定归因完成。
B40同时给定Native Query/cache后输出通过，max_abs=0.0078125；
其主cache差异也会影响输出，与B16的证据分别记录。原始张量及各替换边界见两档
`output_boundary.pt`和`output_boundary.json`，CPU计算仅用于归因，不作A3舍入oracle。

## 62. 2026-09-22：G08跨流生产延迟六组通过

第59节首轮`full_cross_stream_v1`六个任务均在启动shell解析时exit2：
队列截断多行命令参数，单引号未闭合，未执行Python数值检查。
改为结果目录保存`run.sh`并仅提交单行`bash /abs/run.sh`后，
`full_cross_stream_v2`六个任务均exit1：当前torch_npu2.10运行时无私有`npu._sleep`，
虽然安装包测试辅助代码仍引用该接口。保留失败源码与任务记录，未修改torch_npu。

`full_cross_stream_v3`改用公开`torch.mm`，在独立1024×1024 BF16缓冲上做0/8/32次设备计算，
六组全部通过，但部分延迟短，未证明submit返回时生产仍未完成，因此补做更强延迟。
`full_cross_stream_v4`用8192×8192独立缓冲、0/4/16次计算；在原submit返回后立即用
非阻塞Event.query记录生产状态，最大延迟的eager和graph两次提交均要求至少一个生产事件未完成。
耗时读取和延迟缓冲数值检查都在共享重放测试完成同步之后；没有host sleep或热路径取值。
只包装测试进程中的选定stage，保留原Native task.run、ExternalEvent与复用fence，生产executor未改。

六组任务均终态exit0，A→B→A各30项、共540项完整Native/eager/graph检查通过；
源hash与manifest一致，延迟缓冲结果精确正确，最大延迟生产事件均有未完成证据。

| B | Native stage | 任务 | 0次耗时中位数ms | 16次耗时中位数ms |
| --- | --- | --- | --- | --- |
| 4 | COMPRESSOR | task_20260922_155922_221854817973 | 0.12740 | 58.80614 |
| 4 | INDEXER | task_20260922_155922_22185887973 | 0.16531 | 59.05063 |
| 4 | ATTENTION | task_20260922_155922_221866531783 | 0.06888 | 59.26677 |
| 40 | COMPRESSOR | task_20260922_155922_22187588130 | 0.16971 | 58.53497 |
| 40 | INDEXER | task_20260922_155922_221888918286 | 0.32148 | 58.77945 |
| 40 | ATTENTION | task_20260922_155922_221903613391 | 0.25122 | 59.11405 |

证据见`full_cross_stream_v4/summary.json`、各case的`cross_stream.json`及`full_replay.json`。
该结果覆盖参考B4/B40固定BS的G08；未做删除等待的负对照，不扩展为G04～G07、padding、
请求生命周期、prefix共享、正式目标权重或完整P3验收。

## 63. 2026-09-22：先移除已确认的适配冗余，保留剩余方案讨论

用户明确最终CSA应为独立算子，并要求先讨论外部8次适配的必要性，随后授权
“先消除能消除的冗余适配，剩下的再讨论怎么处理”。因此本轮不直接把8次调用整体合并，
也不改两类state窗口的存储策略。第62节G08六项已完成，属于实质验证进展；
本轮从当前源与终态证据继续，没有重启旧任务或更换数值阈值。

当前清理包括：删除没有计算消费者的`window_swa_lens`参数、缓冲和写入；
删除外部`prepare_rope`注册/调用及普通cos/sin转换缓冲；普通RoPE直接引用Native
同一层的FP32交错频率表，保持Native生产与事件等待。两组压缩metadata仍按闭合行展开，
但保持Native列布局，不再先抽取偶数列再由CSA重复交错。
CSA、QKV和inverse-RoPE消费者直接读取交错cos，删除五份内部cos重建缓冲；
保留原前向/逆向sin符号、pair-swap及浮点旋转运算顺序。

适配测试改为逐bit验证实际Native频率与原往返转换结果相同、压缩有效行频率保留原值、
非闭合行仍为cos=1/sin=0；完整比较检查普通频率指针与Native相同。
诊断沿用相同生产函数并更新频率布局说明，旧捕获证据不改写。
修改前完整源与hash保存在`adapter_redundancy_v1/before/`及`before_sources.json`。
CPU完整编译和真机验证结果待追加，不能将改动前18/18与100步PASS直接外推到本版本。

本轮目标调用数为7次外部适配加1次CSA主体；剩余token metadata、两组压缩metadata、
两组state gather/commit的必要性及最终处理方式留待用户讨论，不自行扩大为state直写或整体融合。

CPU token/compact metadata及完整CSA编译（含PTOAS）全部通过，见`compile.json`。
Native适配真机B4/history131071、B40/history131073分别为
`task_20260922_162830_22592921799`、`task_20260922_162830_22657820928`，均exit0 PASS；
普通Native频率与旧往返重排结果逐bit相同，压缩频率有效/无效行及state保护检查通过。
完整单层18组已全部终态exit0，共324项检查通过，并与`reference_matrix_v5`保存的
PTO输出、Top-K和分数逐bit相同；所有普通频率输入指针均直接指向Native缓冲。
两项先导完整比较为`task_20260922_162830_2278921367`（B4/history131071）和
`task_20260922_162830_2285882675`（B40/history131073），其余16项与后续任务见manifest。

同版本已提交B4/B8/B16/B40原100步轨迹，以及B4/B40三种stage的增强G08延迟；
保留原容量和seed，以前后逐步完整allocation指纹检查行为等价。B16/B40原有输出失败
仍属于未解决数值问题，不因本轮去冗余回归而计为P3通过。原始任务和源码hash保存在
`adapter_redundancy_v1/manifest.json`；CPU汇总脚本为该目录`summarize.py`，连续与profiler结果待终态归档。


第63节后续任务现已归档：B4 `task_20260922_163023_185657011536` eager100+graph100通过，
全部200步的PTO输出、Top-K及完整allocation指纹与清理前一致；最后一次graph profiler为
8次Simpler AICPU加8次AICore kernel_mode提交，较清理前各少1次。
B16 `task_20260922_163023_185815630925`、B40 `task_20260922_163023_185889810970`
分别在原step63/[31,3469]和step46/[185,3015]输出失败，64/47步指纹均与旧版一致。
B4/B40三种Native生产stage的六项G08回归全部exit0，共540项检查通过，最大延迟仍有
submit返回时生产未完成证据；任务号见manifest。

用户随后明确“不要做过度测试，继续做其他的冗余适配消除”。据此停止当时仍在运行的
B8 `task_20260922_163023_18576842052`（exit130）；已有eager100+graph91步指纹均与旧版一致，
没有已观察到的数值失败，但不能记为本版完整100步graph通过，也不重提该任务。
终态证据封存于 `adapter_redundancy_v1/sealed_validation_before_v2.json`；
封存时源码hash与v1 manifest一致，后续源修改不反向改变这些历史结果。

## 64. 2026-09-22：移除独立token metadata调用，直接消费Native设备索引

按用户继续去冗余且控制测试范围的要求，移除`prepare_token_metadata`函数、注册及调用。
CSA参数直接引用Native NPU Tensor：positions为INT64[T]，普通KV、主state和Indexer state
的slot分别为INT32[T,2]；普通KV页表为Native INT32[B,容量]，有行padding时仅创建共享存储的view。
Host不读取这些Tensor的内容，不转换成Python基本类型；设备端在原消费者处读取并算地址。
Native metadata生产任务、ExternalEvent及调用前等待不变，图捕获参数地址保持固定。

因此删除INT32 position副本、普通KV线性slot、两份state ring slot、128列SWA物理索引及
虚拟state identity页表。state历史读取直接按request*14+position%14定位原有窗口，
SWA读写在消费者内使用Native页表/slot计算物理地址，保留负page/slot检查。
稀疏计划保留已有compressed候选过滤与八行有效块归约，原始窗口有效性按最多5段物理页计算；
不改变QK、softmax、PV、projection或压缩池化的浮点运算顺序。
两组compact metadata展开及两组state gather/commit仍保留，存储策略留待讨论。
目标调用数为6次外部适配加1次CSA；最终一次独立CSA提交仍未完成。

诊断入口同步到相同Native接口：旧保存的INT32位置在诊断加载阶段转换为INT64；
生产调用没有此转换。完整比较增加positions、三组slot和普通页表的零拷贝指针断言。
`full_replay --profile`仅在现有A→B→A最后一次重放采样，不增加重放次数。
本轮计划只执行B4/history131071、B40/history131073各一次完整对比和B4一次图重放，
沿用冻结阈值及参考checkpoint，不扩展18组矩阵或100步轨迹。

修改前源及hash保存在`adapter_redundancy_v2/before/`与`before_sources.json`。
CPU编译前三次分别发现本仓库新代码的变量同名类型冲突、INDEX直接转FP32不支持、
单行row_max的列主序输出不满足32字节对齐；分别通过改名、INT32中间标量转换、
保留八行归约解决。失败日志为`compile_attempt1/2/3.{json,log}`。
没有修改PyPTO、PTOAS、Simpler、PTO-ISA或pypto-lib；最终编译与真机结果待追加。


CPU完整编译（含PTOAS）已通过：`commit_state_window`、完整CSA、
`diagnose_indexer_compressor`及`diagnose_sparse_attention`，见`compile.json`。
仅提交下列三项真机任务，均终态exit0 PASS；本版41份Python源hash与manifest一致。

| 用例 | 任务 | 结果 |
| --- | --- | --- |
| B4/history131071 full_compare | task_20260922_170537_244501323828 | 18项通过，输出max_abs=0.00390625，RMSE=0.00025180363445542753 |
| B40/history131073 full_compare | task_20260922_170537_244506013404 | 18项通过，输出max_abs=0.0078125，RMSE=0.0003858533164020628 |
| B4 full_replay --profile | task_20260922_170537_244516022092 | A→B→A共90项通过；固定地址与Native ExternalEvent保留 |

两档完整比较的PTO输出、Top-K和分数与v1同配置保存张量逐bit相同；普通RoPE及五个设备索引
参数的零拷贝指针断言全部通过。六个cache/state、精确Top-K及保护区检查沿用冻结门槛。
图重放覆盖history131071/131073和页表交换；graph/eager输出、Top-K及五份完整allocation
逐bit相同，Native数值和全部未写区域检查通过。最后一次重放的profiler确认
7次`simpler_aicpu_kernel_exec`及7次`aicore_kernel_mode`，对应1次CSA主体加6次适配；
最初各9次、v1各8次。这里只证明提交数下降，没有据此宣称延迟或吞吐收益。
汇总见`adapter_redundancy_v2/summary.json`，任务、命令及源码见`manifest.json`和`sources/`。
本轮不扩展矩阵或连续100步，也未处理第60～61节的大batch连续输出问题；P3仍未完成。

剩余适配的讨论边界：两组压缩metadata将Native紧凑闭合行的slot/cos/sin展开到token维，
当前消费者确实使用这些展开结果；取消它需要决定由消费者按闭合行直接索引，还是在
同一个CSA program内部保留展开任务。四次state操作分别读取主/Indexer的8行历史到14行
窗口，并将本步6行写回各自Native物理页；取消窗口需同时处理真实page stride、历史读取
和写后读依赖。将这些任务合入CSA可减少外部提交，但不等于消除了拷贝。
本轮保留上述六次调用，不自行改变这两类方案；目标仍是一次独立CSA提交。


用户再次要求继续后，进一步只读核对两个compact消费者：主compressor在
`compressor_ratio4_cache_write`按token tile读取频率，Indexer在其闭合行pool/RMS和
cache/scale写回阶段使用展开频率/slot；均可考虑只改这些索引入口，保留原计算函数。
Native紧凑行号不能简单使用token//4：请求r的起点start_r=seq_len_r-query_len_r，
前缀为此前各请求的sum(floor(seq_len/4)-floor(start/4))；闭合token的行号还须加
floor((position+1)/4)-floor(start_r/4)-1。建议在设备端计算该映射后直接读取各组
自己的compact slot/cos/sin，并保持非闭合行屏蔽与现有sin符号/舍入顺序。

供后续讨论的顺序为：先去掉两份compact展开（总提交7→5），再把两组state gather/commit
作为内部任务纳入同一个CSA program（总提交5→1），保留现有14行窗口及Native物理页接口。
这可先达成一次独立提交；进一步取消state窗口是另一个涉及存储依赖的优化。
此处是具体方案建议，尚未实施；最新通过源码仍为v2 manifest记录的版本。


## 65. 2026-09-22：两组compact消费者直接读取Native紧凑行

用户明确选择先做“两次compact metadata：消费者直接读取Native紧凑行，删除展开缓冲”。
本轮删除`prepare_compressed_metadata`函数、注册及两次调用；普通/Indexer压缩slot改为直接
传Native INT32[C,2]，各自cos/sin为Native FP32[C,64]共享存储view。保留各自的Native
query_start_loc和seq_lens作为设备Tensor输入，不在host获取闭合行数量或其他请求内容。

新增本仓库inline索引helper：每组按实际query边界计算B个行偏移，写入现有单owner的
`csa_rope_sign`任务，未增加外部调用或独立metadata任务。偏移为
prefix-floor(start/4)-1，闭合token的Native紧凑行号为offset[request]+floor((position+1)/4)。
两组偏移独立，不能因当前内容相同而合用各组metadata；分配只依赖已有Tensor形状。

主compressor及Indexer的原RMS/RoPE消费者只将闭合行频率gather到16行片上tile，
非闭合行为cos=1/sin=0且不读取Native未使用尾部；sin符号在原旋转乘法前于消费者内折叠。
压缩KV、Indexer K及FP16 scale写回在原写回任务中直接读取紧凑slot，并保留负page/offset检查。
原矩阵乘、pool、norm、Hadamard、量化及旋转浮点顺序不变。
删除两份token级INT64 slot、四份token级FP32频率及两份token级signed-sin GM缓冲；
仅新增两份INT32[B]行偏移，无T级metadata展开。
四次state gather/commit及14行窗口仍按上轮版本保留，本轮目标为1次CSA加4次适配。

诊断继续复用生产函数，更新到Native compact ABI和同一设备行偏移helper；
完整对比增加六个compact Tensor及三份请求metadata的零拷贝指针断言。
原展开适配单测入口删除对已移除kernel的调用；本轮用完整链对比和图重放覆盖实际消费者。
修改前41份源码/hash存于`adapter_redundancy_v3/before/`及`before_sources.json`。
首次CPU编译发现索引算术结果写INT32偏移时缺少显式cast，已在本仓库补齐，
失败记录为`compile_attempt1.{json,log}`。未修改依赖仓库。
验证只计划B4/history131071、B40/history131073各一次完整对比及B4一次A→B→A图重放；
profile复用最后一次重放，预期总提交由7降到5，最终结果待追加，不预记PASS。


完整CSA及`diagnose_indexer_compressor`的CPU编译（含PTOAS）已通过，见`compile.json`。
生成的orchestration直接把Native compact Tensor传给原RMS/RoPE与cache写回任务，
新增GM metadata仅`cmp_row_offsets`和`idx_row_offsets`两个INT32[B]数组；
消费者内的gather落到片上tile。对应4份生成C++及hash已保存到`generated/`和`generated_sources.json`。

仅提交计划内三项真机任务，均终态exit0 PASS；42份Python源与manifest完全一致。

| 用例 | 任务 | 结果 |
| --- | --- | --- |
| B4/history131071 full_compare | task_20260922_172841_362287212781 | 18项通过，输出max_abs=0.00390625，RMSE=0.00025180363445542753 |
| B40/history131073 full_compare | task_20260922_172841_362447617471 | 18项通过，输出max_abs=0.0078125，RMSE=0.0003858533164020628 |
| B4 full_replay --profile | task_20260922_172841_362615420148 | A→B→A共90项通过；history131071/131073、页表交换、固定地址和Native ExternalEvent |

两档PTO输出、Top-K和分数均与v2同配置保存张量逐bit相同；六个cache/state、精确Top-K、
Native/PTO保护区均沿用冻结门槛通过。普通RoPE/原始索引及新增九个compact/request参数的
零拷贝指针断言全部通过。图重放的graph/eager输出、Top-K及五份完整allocation逐bit相同；
Native数值与三套未写区域检查继续通过。
最后一次graph profiler为5次`simpler_aicpu_kernel_exec`及5次`aicore_kernel_mode`，
确认两次外部compact提交消失，总调用由7降到5（1次CSA、2次state gather、2次state commit）。
没有据提交数变化宣称延迟或吞吐收益。

完整汇总为`adapter_redundancy_v3/summary.json`，源码/命令/任务见`manifest.json`与`sources/`。
本次未追加18组、连续100步或其他NPU测试；第60～61节的大batch连续输出问题仍未解决，
P3不因本轮126项回归通过而整体通过。四次state适配及14行窗口留待后续单独讨论和处理。


## 66. 2026-09-22：直接读写两组Native state，删除四次适配与14行搬运窗口

用户明确要求直接读写Native state，同时消除四次适配和两份14行搬运窗口；若无法合理
实现则先讨论。本轮在本仓库实现直接存储访问，未修改PyPTO、Simpler、pypto-lib或PTO-ISA。
主compressor与Indexer分别接收各自Native FP32 state的零拷贝物理页view和Native页表。
view形状为[物理页数, 实际页间距对应的FP32元素数]，保留原storage offset、页padding及owner；
host只查看Tensor的shape/stride等描述，不读取position、slot或页表内容。

两个原pool任务按逻辑position查各自Native state页表，读取两token页内的value/score。
本步token继续使用同一projection结果叠加APE，保留原pool浮点运算顺序。负逻辑位置继续
使用屏蔽score；有效逻辑位置但负page时保留旧gather的全零行语义。原state写回任务直接
按各自Native [page, offset] slot写入对应页，仅写本步有效行，不触碰页padding和未写区域。
写回保留对对应pool任务的显式依赖；生成orchestration中读操作为add_input，写操作为
add_inout，根CSA两组state均声明为InOut。请求间写入隔离仍遵循Native自身存储约束。

删除gather_state_window/commit_state_window函数、注册和两组外部调用，删除native_metadata.py；
NativeCSACall.__call__只调用一次完整CSA。两份[batch*7,2,width] FP32窗口及其容量常量删除，
每请求消除14*(2048+512)*4=143360字节，B40为5734400字节。该数仅为窗口分配减少量，
不计其他内部计算scratch，也不据此宣称延迟或吞吐收益。
Indexer原有8行pool计算暂存继续服务既有算法，不是被删除的14行state搬运窗口。

诊断继续复用生产project/pool/write函数，输入改为Native物理state和页表。
完整比较新增两组state及页表的四项零拷贝指针断言；原适配探针改为只检查Native描述符，
实际读写由完整CSA对比、图重放和连续轨迹验证。首次CPU检查发现探针import缩进错误，
已修正，失败记录保存在adapter_redundancy_v4/compile_attempt1.{json,log}。
完整CSA与共享Indexer诊断的CPU编译（含PTOAS）随后通过；生成的两组pool、write及
orchestration共5份C++和hash保存在generated/及generated_sources.json。

证据根目录为results/cann90_20260921/adapter_redundancy_v4/；修改前42份Python源保存于before/，
验证版本为sources/中的41份源码及一项删除记录，详见manifest.json。
仅提交四项针对性NPU任务：B4/history131071、B40/history131073完整比较，B4 A→B→A图重放
并采样最后一次profile，以及B4 mixed10（eager10+graph10）。mixed沿用[1,6,2,5,5]接受模式，
覆盖拒绝后继续和全部接受；不将短轨迹视为旧100步失败的修复证据。结果待追加，不预记PASS。

上述四项真机任务均已终态exit0 PASS；41份Python源码hash匹配manifest，已删除文件确认不存在。

| 用例 | 任务 | 结果 |
| --- | --- | --- |
| B4/history131071 full_compare | task_20260922_174951_14162671305 | 18项通过，输出max_abs=0.00390625，RMSE=0.00025180363445542753 |
| B40/history131073 full_compare | task_20260922_174951_141630714069 | 18项通过，输出max_abs=0.0078125，RMSE=0.0003858533164020628 |
| B4 full_replay --profile | task_20260922_174951_141634731404 | A→B→A共90项通过；history131071/131073、页表交换、固定地址和Native ExternalEvent |
| B4 mixed10 continuous | task_20260922_174951_141641731666 | eager10+graph10共370项通过，平均推进3.8，Native接受修正与跨步状态保留 |

两档完整比较的PTO输出、Top-K和分数与v3同配置张量逐bit相同，新增两组state及页表
零拷贝断言全部通过。六个cache/state、精确Top-K、Native/PTO保护区在冻结门槛下通过。
图重放和连续轨迹均确认graph/eager输出、Top-K及五份完整物理allocation逐bit一致。
图重放最后一次及连续轨迹最后一步的两份profiler分别只有1次simpler_aicpu_kernel_exec
和1次aicore_kernel_mode，确认PTO外部提交由5次降为一次完整CSA，未残留state搬运调用。
Native metadata生产任务和ExternalEvent等待仍属于原调用方，不计作PTO适配调用。

汇总为adapter_redundancy_v4/summary.json（PASS，496项）；profile路径、逐bit结果及任务终态
见该汇总和queue_snapshot.json，命令与源码见manifest.json/sources/。本次只执行上述四项
NPU任务，没有重跑18组或100步矩阵。原B16/24/32 step63、B40 step46输出问题仍待处理，
P3整体、服务forward派发、正式ModelSlim权重验收与P5均未因本轮适配消除而完成。


## 67. 2026-09-22：Native/PTO PyTorch profiling对比与单次CSA泳道图

用户要求提供两份对比的PyTorch profiling和PTO泳道图，并参考Qwen仓库方法。
只读参考Qwen的qwen3_aclgraph_profile.py、qwen3_single_layer_swimlane.py和对比脚本：
Native/PTO分开进程采集ACL Graph；DFX另开eager进程，只包围一次完整PTO CSA调用。
本仓库新增dsv4_csa_profile.py、run_csa_profiles.sh和compare_dsv4_csa_profiles.py，
复用现有Native attention、Native metadata/存储fixture及生产NativeCSACall；未修改计算源码或依赖仓库。

条件为B40/S6/history131073、TP1、ND、model.layers.2.attn、seed1024，使用48分片参考权重。
三个独立进程在同一队列任务内依次使用logical device8；任务task_20260922_185453_15838428498
终态exit0。每个进程先运行2次eager预热；profiling两组再capture同一完整CSA图并预热重放1次，
正式各采3次重放。使用torch_npu PyTorch profiler，CPU+NPU、Level1，关闭stack/shape/memory。
Level1包含AscendCL图执行API；此前v4的Level0 profile仅用于提交数断言，不混入这次对比。

每次profile marker包括Native metadata submit、graph replay和完成同步；核函数device span
按marker内所有设备任务的最早start到最晚end统计，包含metadata生产stream及CSA，
不包含权重加载、编译、初始化或预热。本次为固定输入单层采样，不是服务或连续接受轨迹性能。
两组各确认3次AscendCL@aclmdlRIExecuteAsync；PTO每次只有1个Simpler AICPU和1个AICore入口。
输入hidden/position、全部5份初始allocation、5组页表、权重记录和配置逐项一致；
各自profile输出与预热eager逐bit一致，Native/PTO输出按冻结门槛通过，
max_abs=0.0078125、RMSE=0.0003858533164020628。

| profile | 三次device span（微秒） | 中位数（微秒） |
| --- | --- | --- |
| Native | 3010.64 / 2473.26 / 2432.02 | 2473.26 |
| PTO | 56087.22 / 31145.62 / 31018.80 | 31145.62 |

当前这组三次采样的PTO device span为Native的12.593倍；首样本开销单独保留，未删除或补采。
该结果说明单次提交已完成，但计算/调度性能仍需分析；本轮仅交付profile，不做算法或性能改动。

DFX进程使用enable_chip_swimlane=4、enable_dep_gen=True；Native metadata就绪后，
在begin_dfx/end_dfx之间调用一次生产CSA。原始run_boundaries恰好1个、dropped为0，
1265条原始AICore任务记录经Simpler官方swimlane_converter转为2549个设备任务切片。
保留chip_swimlane_records.json和deps.json，导出merged_swimlane.json；该文件含任务泳道及依赖。
DFX输出与PTO graph输出逐bit相同。DFX边界同步且带诊断开销，不参与上述ACL Graph耗时对比。

全部结果在results/cann90_20260921/csa_profile_v1/：native/native_profiling.json、
pypto/pypto_profiling.json、swimlane/merged_swimlane.json为三个可直接打开的产物；
comparison.json为条件核对、逐次统计和输出对照，manifest.json保存命令、源hash和队列任务。
本次只提交一个任务、三个必要采集进程，未扩大精度回归或重跑100步。

首次converter导出的依赖和任务完整，但标签只有func_id。已从本次DFX实际编译目录
build_output/_jit__decode_csa_tp1_attention_eqn4j_g9/kernel_config.py提取45个函数ID→名称，
保存kernel_config_source.py及name_map.json，核对deps中所有有效kernel_id均有映射。
用官方converter的--func-names离线重导出；2549个设备任务切片均已有真实名称，
4220对依赖flow保留，补名称前后的ph/pid/tid/ts/dur逐条一致。此步骤未重新上卡。
阅读说明与复现命令保存于结果目录README.md；三个trace及原始DFX、依赖、源码和对比记录
另打包为DSV4_CSA_B40_S6_H131073_profiles.tar.gz，包外SHA256见同名.sha256文件。

## 68. 2026-09-22：Indexer 按 Native 整页加载，消除小 DMA 热点

用户指出泳道图Indexer异常后，核对实际Worker View和生成代码：
`indexer_score_topk_leaf_aic`的24个block平均执行28173.50μs，AIV的48个block
平均28170.78μs；`indexer_topk_query_merge`自身仅35.96μs，长条主要来自前置依赖等待。
当前Native共享页stride4160字节，包含4096字节INT8 K与64字节FP16 scale。
此前本仓库每页用32次`gather_row([1,128])`适配，384-token score tile需要384次小读取；
参考库独立连续K布局可用`[32,128]`一次读取一页。

第一版页内slice后reshape成`[32,128]`再作GM源，CPU lowering拒绝：slice已被转成Tile，
`tile.gather_row`要求GM Tensor。失败日志归档在`indexer_page_load_v1/failed_gm_slice/`，
未执行NPU，不修改编译器。
最终在原score leaf中用两个AIV各读取6页，每页一次`[1,4096]`到片上24KiB UB，
reshape为`[192,128]`后经现有`aic_gather(UP_DOWN)`交给Cube。每个score tile的K
GM读取次数从384降至12，读取总字节数仍为49152，增加一次UB→L1片内传递。
原始页表、真实page stride、非零storage offset、尾页clamp和共享K/scale allocation继续使用。
无新增GM搬运缓冲、CSA根参数或外部适配调用；INT32点积、FP16转换、系数归约和Top-K
tie顺序保持不变。生产改动只有本仓库`decode_indexer.py`；PyPTO/Simpler/pypto-lib/PTO-ISA未改。

CPU lowering与完整PTOAS编译通过。B4/history131071
`task_20260922_200911_325781123742`、B40/history131073
`task_20260922_200912_325787534`均exit=0，合计36项完整比较PASS；
输出、Top-K、scores与`adapter_redundancy_v4`逐bit一致，六项cache/state及两侧保护区通过。
B4输出max_abs=0.00390625，B40=0.0078125；冻结阈值未变。

同卡8的新Native/PTO profile与独立eager DFX任务
`task_20260922_201208_404184422233`完成exit=0。B40/S6/history131073、seed1024，
同一参考checkpoint/layer/初始输入与状态哈希；三个进程输出分别与旧profile逐bit相同。
每次graph重放仍为一次PTO提交，DFX窗口也只有一次完整CSA。

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| score leaf AIC平均执行，DFX | 28173.50μs | 5701.32μs |
| score leaf AIV平均执行，DFX | 28170.78μs | 5706.04μs |
| Top-K query merge平均执行，DFX | 35.96μs | 35.20μs |
| 完整CSA加Native metadata，graph profile三次中位数 | 31145.62μs | 13522.88μs |
| 同条件Native三次中位数 | 2473.26μs | 2479.02μs |

Indexer AIC约4.94倍加速，完整CSA中位数约2.30倍，当前PTO仍为Native中位数5.45倍。
新版PTO三次跨度14332.70/13522.88/8307.74μs，旧版56087.22/31145.62/31018.80μs，
全部保留；DFX独立eager窗口与graph profiler时间不混算。
新泳道图使用实际`_jit__decode_csa_tp1_attention_i668rr27/kernel_config.py`映射，
1265个AICore block、2549个设备切片和原始依赖保留，补名称前后所有时间戳一致。

离线比较初次因三个derived-zero weight-offset记录的set迭代顺序不同而拒绝。
逐名称核对shape/dtype/SHA256/loaded_exact/source全部相同后，比较工具改为按唯一名称
比较完整记录并拒绝重名；原始metadata不修改，无需重新上卡。
执行源码43份哈希与捕获时一致，更新后的离线工具独立记录在postprocessing_sources。
证据、复现命令、新旧图入口及下载包见
`results/cann90_20260921/indexer_page_load_v1/README.md`，汇总`summary.json`为PASS。
本次只验证参考层B4/B40和B40固定输入profile，不扩展100步测试；未变更完整P3及目标权重验收状态。

## 69. 2026-09-22：Indexer页表预取无实测收益，撤回本轮尝试

继续优化时先尝试将score tile从384扩大到448，CPU编译报告Vec占用204800字节，
超过188416字节上限；失败源码与日志保存在indexer_prefetch_v2/tile448_compile_rejected/，未上卡。
恢复384后，在每个score leaf循环外预取最多256个Native页号到1KiB UB，
K与scale读取复用该页表片段。完整CPU lowering/PTOAS通过；生成代码确认一次页表加载，
循环体内12次页号读取来自UB。未修改PyPTO、Simpler、PTO-ISA或pypto-lib。

B4/history131071任务task_20260922_203644_354637125208与B40/history131073任务
task_20260922_203644_354641931578均exit0，共36项完整比较通过；输出、Top-K、scores
与已验证整页加载版本逐bit相同。冻结阈值、cache/state及保护区检查保持不变。

同卡8任务task_20260922_203916_375661432689完成exit0，分别以独立进程加载修改前后
冻结源码，各采5次graph重放，另采Native与一次独立DFX。源码来源及输入条件核对通过。
修改前设备跨度为[9192.18,8448.60,8470.40,13521.12,8363.44]微秒；
预取版本为[9208.92,8456.94,13695.96,8613.32,8434.52]微秒。
中位数8470.40→8613.32微秒，本次增加1.69%；Native中位数2485.78微秒。
独立DFX的Indexer AIC均值为6031.48微秒，上轮诊断为5701.32微秒。
五次样本不足以认定稳定回退幅度，但没有测出保留预取改动的性能收益。

因此已精确恢复整页加载v1的decode_indexer.py，SHA256为
469ff5dc61d35a4692dc6650ed8498d151c2ac8526b59ddf9f4eab11d5132267；
未重复上卡。尝试版本、生成代码、原始trace及逐项对照保存在indexer_prefetch_v2/，
summary.json的PASS仅表示数值和证据一致性，撤回决定见decision.json。
离线profile工具支持显式--source-snapshot核对历史冻结源码，避免撤回后误用当前代码核对。
本轮没有新增已确认的性能优化，保留第68节的整页加载优化。

## 70. 2026-09-22：对照计划核对剩余工作

P1完整通过，P0/P2/P3/P4部分完成，P5尚未启动。剩余工作归为七类：
服务forward接入、P3大batch连续输出精度、P3剩余场景、P4完整双卡协调、
正式权重P0/P2、P5六项整模型验收、性能与同步验收。
详细缺口已写入计划第10.1节；第9节过时的“adapter/compare/graph尚未实现”描述已修正。
独立CSA及八次外部适配消除已经完成，不重复列入待办；此次核对未重跑测试。

## 71. 2026-09-22：暂停P3场景/P4，修复连续精度中的softmax概率舍入

按用户最新指示暂停padding及P3剩余场景、P4完整双卡工作，先处理第60节四档连续精度失败。
没有修改PyPTO/Simpler/PTO-ISA/pypto-lib、Native算法或冻结阈值。

复用生产sparse_attn_csa及sparse_attn_csa_tp1，将历史B16 query31和B40 query185所选
128+512行KV映射到小缓存，复制24份相同query。任务
`task_20260922_214738_15105076164`、`task_20260922_214738_151177416009`
均exit0；各786432个projection-input元素与原失败现场逐bit一致，24份分子/分母/max也一致。
这是失败算术的隔离复现，不是连续状态轨迹验收。证据：`sparse_boundary_v2/`。

B16 head51分子误差可由第456个selected KV乘以-1/256解释，残差RMSE约6.37e-7；
用设备分子/分母做CPU除法仍复现全部91个NoPE差异，排除最终除法为这组差异的主因。
Native及PTO生成代码的QK均顺序累加4个K=128块，未发现分块大小不同。
随后小型QK设备任务`task_20260922_215842_34208492236`和
`task_20260922_215842_342171915649`均exit0；保存分数重建的max与生产max相同，
CPU后处理可复现两例目标head的PTO NoPE结果。CPU仅用于归因，不作为A3精度oracle。
首轮`sparse_scores_v1`两个任务因诊断spmd未读取block index而编译失败，修正后为v2，
没有把失败任务记作数值通过。

直接原因是Native `sparse_attn_sharedkv_scfa_block_vector.h:482`发布BF16概率使用
`CAST_ROUND`（中点远离零），本仓库PTO使用`rint`（中点取偶）。Native最终BF16输出
在同文件Bmm2CastAndCopyOut使用CAST_RINT，不能把两处转换混为一谈。
B16 head51/selected456的概率为0.595703125：rint得到0.59375，round得到0.59765625。
B40 Native输入head34/selected292也命中中点0.20947265625，对应0.208984375/0.2099609375。
源码只改softmax概率cast为mode="round"，保留最终输出cast、QK/PV、归约和阈值。

候选小复现`sparse_round_v1`：B16任务`task_20260922_220145_179229812190`，
使用Native Query/cache后目标query最终4096列输出与Native逐bit相同，24份复制均一致；
attention projection输入从101个BF16差异降至1个（head49/col95），不是宣称中间结果全相同。
B40任务`task_20260922_220145_179236221436`使用原PTO Query/cache仍失败，
原col3015超差保留，max_abs=0.0106201171875。该轮脚本进程exit0但JSON明确FAIL，
已补充失败assert，后续候选数值失败会返回非零；不能以exit0覆盖数值FAIL。

进一步检查B40 query185：Query逐bit相同，512行压缩KV完全相同，原始128行SWA有12个
BF16差异，涉及selected行5/14/44/59/83/117/124/127；因此当前这一例须沿SWA KV链定位，
不能仅因主compressor state非bitwise便归因于压缩KV。按原seed及接受轨迹恢复这些行的
8份原始输入，来源step14/17/24/29/35/44/46/46，保存在`swa_boundary_v1/`。
KV投影/RMS设备小诊断任务`task_20260922_220846_390155913187`已提交，结果待归档。

B16/24/32原mixed100并行任务分别为`task_20260922_220306_21198607658`、
`task_20260922_220306_212069431879`、`task_20260922_220306_212145012473`，
保留history131071、seed1024、100步及fixture容量131683，记录在`continuous_round_v1/`。
提交时不预记PASS；B40已知仍失败，未盲目重跑其100步。均属于参考checkpoint层级验证，
正式ModelSlim权重、完整P3/P4/P5状态不扩大。

SWA后续：首次诊断因Python局部名signed生成C++保留字而编译失败，仅将诊断变量改名，
未改编译器。`task_20260922_221017_39544921029`完成exit0，生产KV小复现的8行NoPE
与旧PTO逐bit相同；给定Native BF16投影后，通过精确identity matmul继续调用同一生产
KV RMS函数，全部8×512值与独立Native RMS一致，线索缩小到投影。
该初版Native矩阵乘用了32行/KN连续权重，3行与历史Native不完全相同，因此不能拿它
替代旧现场的Native投影oracle。下一版改为原始240行输入、原权重stride及实际
Native unquantized_gemm使用的F.linear；8行NoPE全部复现历史Native。

`task_20260922_221346_12638207247`仅在独立诊断进程将KV_OK从2设为1，
调用同一生产KV函数验证单条K累加链，不改生产源。结果9个NoPE差异减少为3个，
仍在selected5/124/127各1个；给定Native投影的RMS仍全部一致。
这只是算术归因，尚未验证完整B40输出及轨迹，不保留该生产改动。
证据为`swa_boundary_v1/single_k/swa_boundary.json`，其中PASS表示诊断执行完成，
各项bitwise比较单独记录，不能解释为B40精度PASS。
复现输入脚本`swa_boundary_v1/reconstruct_inputs.py`和舍入CPU归因脚本
`sparse_scores_v2/analyze_rounding.py`一并保存；当前生产代码仍只有概率cast这一处数值修改。

三档mixed100现已全部完成：B16/B24/B32各eager100+graph100，分别3700项、合计11100项
数值检查全部PASS；逐步output/Top-K及五份完整allocation的graph/eager指纹完全相同。
原step63超差均消除，后续至step99继续满足冻结门槛，六项cache/state与Native/PTO保护区通过。
完整逐项复核、原任务日志及数值汇总见`continuous_round_v1/summary.json`。
这确认上述三个batch的参考mixed轨迹修复，不表示中间张量全部与Native逐bit相同，
也不将B40或完整P3预记通过。

执行状态需独立说明：三个Python驱动均写出最终PASS并完成清理后，启动bash因运行期间
在同一脚本插入swa_boundary case、后续文件读取偏移改变而报unexpected EOF，最终exit2。
因此任务进程不是exit0；不能只读取队列退出码抹掉完整数值证据，也不能将其称为执行全程PASS。
生产14份源码/连续驱动哈希与提交manifest相同，全部600步的11100项检查逐项复核通过；
当前启动脚本bash -n通过。本轮不为收尾shell错误重跑已完成的600步。
原启动脚本已重建归档，并保存绝对workspace路径的`run_frozen.sh`供后续独立提交，
避免运行期间修改共享启动文件。详见`execution_note.json`；未修改队列或其他任务。

## 72. 2026-09-22：B40 SWA KV投影的Native K遍历顺序

继续用户指定的B40差异排查，未恢复P3其他场景或P4工作。复用既有QA排查中的本机CANN
MatMulV2生成器，以真实WKV的NT输入形状M×4096、512×4096生成六档CPU代码，
均完成；源码和可复跑脚本保存在`kv_native_codegen_six/`。不修改CANN及任何依赖仓库。

M240的生成代码给出：N按96列分组，前四组g=0..3的K256块访问顺序是
`(5*(g//2)+(k if g%2==0 else 15-k))%16`，k=0..15；后两个重叠尾组保持正序。
每个K256块内按四个K64顺序Mmad累加，整个K4096使用同一FP32累加器。
这与原PTO两段K2048 split-K、所有输出列统一正序的数值顺序不同；不是硬件不支持，
也不属于PyPTO/PTOAS/PTO-ISA精度选项失效。其他M的Native分块不同，未由B40外推。

新诊断`dsv4_csa_swa_window.py`按原seed1024、接受轨迹和最终写入者恢复全部128行
SWA的输入，覆盖原step13～46，逐步仍使用原M240。Native复用F.linear、npu_rms_norm
和inplace_partial_rotary_mul；PTO复用同一生产kv_proj_rope，RoPE也核对捕获值。
基线任务`task_20260922_224756_4792222156`完成exit0：Native和PTO各65536个元素
分别与失败现场逐bit相同，精确复现原12个BF16差异，排除小矩阵形状替换导致的假归因。
证据：`swa_window_v1/baseline/swa_window.json`。

在本仓库qkv_proj_rope.py只对M240启用上述Native累加顺序：N32避免跨96列边界，
K64匹配实际Mmad，正反向仅作用于K256块序，块内顺序不反转。所有K块共用一个累加器，
仍在同一PTO CSA提交内执行。其余token数继续原投影路径，RMSNorm、RoPE、量化边界、
cache写入及冻结阈值不变；逐文本核对见`source_scope.json`。
KV诊断和完整CSA CPU编译（含PTOAS）均通过，未修改PyPTO/Simpler/pypto-lib。

候选任务`task_20260922_225047_996549771`完成exit0：完整128×512 SWA与Native
逐bit相同，原12个差异归零。把全部重算的128行接回原PTO Query及512行压缩KV，
复用生产attention/O-proj，失败query185的最终4096列输出与Native逐bit相同，
24份复制全部相同，max_abs=0。没有只替换超差坐标或写入Native黄金值。
证据：`swa_window_v1/candidate/swa_window.json`及`candidate/attention/sparse_boundary.json`。

B40原mixed100复验已提交为`task_20260922_225215_171720226167`，
history131071、seed1024、S6、fixture容量131683及全部阈值不变，eager/graph连续保留状态。
任务、冻结源及独立启动脚本见`continuous_kv_native_b40_v1/manifest.json`。
启动脚本在提交前冻结，本次不修改运行中的脚本。

该任务完成exit0：B40原mixed100的eager100和graph100全部通过，分别1800、1900项，
合计3700项检查无失败。原step46完整240×4096输出max_abs=0.0078125、
RMSE=0.00039212923729792237、超差0；两个phase结果相同。全部200步的输出、
Top-K、六份cache/state及Native/PTO保护区均通过；每步graph/eager输出、Top-K
和五份完整allocation指纹逐bit一致。全轨迹输出最大绝对差0.01171875，仍满足已冻结的
逐元素`abs_error <= 0.01 + 0.01 * abs(reference)`，未调整阈值。
45份源码及启动脚本哈希与提交manifest一致。证据：
`continuous_kv_native_b40_v1/b40/continuous.json`、`summary.json`及`task.log`。

至此用户指定的B16/24/32 step63、B40 step46参考mixed100输出超差已修复并有完整
eager/graph数值复验；前三组shell收尾exit2仍按第71节单独记录。本次不把失败query
局部逐bit相同外推为全层Native/PTO逐bit相同，不计作正式目标权重验收或完整P3通过，
也未恢复已暂停的P3其他场景和P4。

## 73. 2026-09-22：继续缩小B40 step46容差内输出差异

用户追加要求尝试降低第72节step46的max_abs=0.0078125。该目标是继续缩小容差内
浮点差异；此前冻结容差PASS仍成立，不能将其描述为Native/PTO逐bit相同。

沿用同一连续驱动和生产函数，增加仅用于诊断的`--stop-after-step`：保留`--steps 100`
决定的fixture容量131683，仅执行原eager step0～46。诊断结束状态明确为
`DIAGNOSTIC_COMPLETE`，不计作100步或graph验收。OutputBoundary已有分段对照覆盖
Native/PTO Query、两种cache、attention与O-proj，本次补存hidden/positions/Top-K便于复现。
冻结源码、启动脚本及命令见`step46_residual_v1/manifest.json`，任务为
`task_20260922_233223_877510839`。提交后16卡均被另一项整模型任务占用，当前仍排队，
尚未获得本轮step46的最大差异坐标；没有借用历史query185坐标冒充当前残差位置。

等待期间，对既有B40 step46捕获的Native WO-B INT8输入、权重和scale做CPU精确整数点积：
全部240×4096输出中，`(acc*activation_scale)*weight_scale`有8处BF16差异，
最大0.0009765625；`(acc*weight_scale)*activation_scale`与Native全部逐bit相同；
`acc*(activation_scale*weight_scale)`有11处差异、最大0.001953125。
本机CANN的`quant_batch_matmul_v3_pertoken.h`也有先AscendDequant通道scale、再乘
pertoken scale的实现，源码摘录/哈希及CPU结果分别见`wo_b_source_evidence.json`、
`wo_b_dequant_all_cpu.json`。这条线索不能解释或证明消除本轮0.0078125最大差异，
不能直接外推Indexer的尺度合并顺序。

准备CPU边界分析入口`step46_residual_v1/analyze_capture.py`，保存最大差异坐标、
各替换边界输出及对应Query/SWA/压缩KV差异。此轮生产CSA源码未变，未改依赖、Native
或冻结阈值；当前状态为等待设备采集，不预记残差降低或精度修复完成。

## 74. 2026-09-23：B40 step46残差分解为Query尺度顺序和主compressor状态投影

第73节采集任务已完成exit0，原100步容量、seed与状态轨迹不变，只执行eager step0～46，
状态为DIAGNOSTIC_COMPLETE。重放生产边界与整层PTO输出逐bit相同。
max_abs=0.0078125对应四个坐标：`[12,1408]`、`[109,322]`、`[109,2394]`、`[224,3395]`。
query12的Query和128行SWA均逐bit一致，仅selected146（压缩索引32807）的col317/456
有差异；换Native cache后该query全部projection input逐bit相同。
query109的Query有6个BF16差异，query224有1个，二者换Native Query后目标输出坐标一致。
详细替换边界及数值见`step46_residual_v1/current_attribution.json`。

Native QR到Query独立任务`task_20260922_235806_84780428262`完成exit0：保留原240行，
用加载后BF16再转FP32的WQ-B scale复现Native npu_quant_matmul、RMS及RoPE，Native Query
与捕获全量逐bit相同。CPU精确整数点积对比反量化顺序：先激活后权重有40个BF16投影差异、
85个最终Query差异；先权重后激活仍有30/63个；先合并scale则投影与Query全部逐bit相同。
首次任务`task_20260922_235625_386634317292`误用checkpoint原FP32 scale，无法复现Native，
已记录失败并修正诊断加载；没有调整Native加载规则，也不拿失败结果作归因。

生产q_proj_q_dequant的完整tile和tail均改为先合并两尺度，再乘INT32转FP32的累加值。
PTO小复验`task_20260923_000812_200772920363`完成exit0：query12/109/224均与Native
逐bit相同，全240行Query剩6个其他BF16差异；未宣称整个Query全量逐bit相同。
两次诊断编译修正分别处理广播列维度和tail的Tensor/Tile层级，未改编译器。
证据：`query_dequant/query_boundary.json`和`query_pto_candidate/query_boundary.json`。

query12的压缩索引32807最后由step42/token13写入。原容量连续采集任务
`task_20260922_235929_93084129834`完成exit0，保存主compressor的Native/PTO前后state、
全部当前输入和Native fused compressor输出。小重放基线
`task_20260923_000520_119280813104`完成exit0：64×512个本步cache值逐bit复现原PTO，
与Native有15处差异；仅换入Native前态即消除query12的两处差异。Native当前FP32投影与
PTO有约21万处末位差，给定Native当前投影及前态后cache仅剩2处其他差异。
首次小重放任务因诊断INT32 slot赋值用了INT64 arange而失败，修正后才接受基线证据。

只读Native compressor_kernel_perf.h与compressor_block_cube_perf.h确认当前均匀S6分支：
每512列分16个N32组，组g从K=g×256开始循环，L0 Mmad K128。原PTO是N64/K512且统一
从K0开始。生产main compressor改为N32/K128及上述K起点，保留首块pl.matmul以维持
尾部48行的compact accumulator metadata，并按kb==0初始化；没有改pool、RMS或state布局。
候选小任务`task_20260923_000812_200777614565`完成exit0，240×1024个value投影和
加APE后的score投影均与Native FP32逐bit相同。证据：`main_replay_candidate/replay.json`，
源码依据及哈希见`main_projection_source_evidence.json`。

完整CSA CPU编译通过。两处修正合并后，原容量B40 step0～46复验任务
`task_20260923_000902_203106929095`与B4完整同图A→B→A任务
`task_20260923_000952_205293316858`已提交，结果待归档；未预记step46最大差异降低。
冻结源和独立启动脚本位于`step46_residual_v1/candidate_continuous/`。
本轮没有修改O-proj，因为已定位的四个最大差异坐标不由其反量化顺序引起。

## 75. 2026-09-23：主compressor八行池化和512列RMS顺序

第74节的B40任务`task_20260923_000902_203106929095`完成exit0，eager step0～46全部
846项检查通过；B4同图A→B→A任务`task_20260923_000952_205293316858`完成exit0，
90项通过。B40的主compressor state在全部47步均与Native FP32逐bit相同。
step46原四个最大差异坐标已消失，输出不同元素由173926降至85157，RMSE由
0.0003921292373降至0.0002592211240；但max_abs仍为0.0078125，位置改为`[224,3770]`，
Native=1.078125、PTO=1.0703125。Query及SWA完全相同，selected压缩KV仅两处末位差，
换Native cache后目标输出恢复一致。证据：`step46_residual_v1/candidate_attribution.json`。
这轮改善不能写作最大绝对差已降低，且47步诊断不是100步验收。

继续只读核对Native compressor的ColumnSoftMax、ColumnSum、RowSum和RmsNorm：
八行窗口按前/后ratio4交错，统一最大值、Exp、8→4→2→1列和；先除概率总和，再乘value，
再以同样树形归约。原main PTO是逐行在线softmax并在最后除分母，数学等价但FP32顺序不同。
RMS则先将512列平方按256、128、64列对半折叠，再对64列WholeReduceSum；最后是
逐行除以sqrt，再乘gamma。原PTO是各64列先求和再累加，以及乘倒数。

在本仓库main compressor中对齐上述两段顺序，未改变投影、Native state布局或写回语义。
八行value/score通过pl.load直接读取Native历史state及当前投影，组装到核内Vec Tile后归约，
没有新增GM搬运窗口或外部PTO调用。CPU lowering生成的scatter_softmax_pool.cpp确认
临时窗口为`Tile<Vec,float,8,512>`；不是重新引入已删除的14行state适配缓冲。

小重放`task_20260923_001546_217947530122`完成exit0：给定Native前态时，所有240行
当前投影与Native FP32逐bit相同，64×512个本步压缩KV也与Native BF16逐bit相同；
前一版给定同样前态尚有2处差异。用捕获的旧PTO前态仍有10处差异，符合旧投影误差
仍留在该输入快照中的事实，不能拿旧前态检查代替从初态连续执行新投影。
证据：`step46_residual_v1/main_replay_pool_native/replay.json`。

局部及完整CSA CPU编译通过。最终从原初态重跑B40 step0～46的任务为
`task_20260923_001643_220160128672`，B4同图A→B→A为
`task_20260923_001643_220164122410`；冻结源码与命令见
`step46_residual_v1/pool_native_continuous/manifest.json`。终态与数值结果如下。

两项最终任务均完成exit0。B40原fixture的eager step0～46全部846项通过，B4完整
同图A→B→A全部90项通过，合计936项。第46步输出max_abs由0.0078125降为0.00390625，
RMSE由0.00039212923729792237降为0.00022528677072841674，不同BF16元素由173926
降为65485，超出冻结容差的元素仍为0。原四个最大差异坐标以及中间版新增的[224,3770]
均恢复与Native相同。基线与最终采集的Native输出、hidden、positions逐bit相同。

主compressor的完整state和压缩KV在全部47步均与Native逐bit一致；Top-K、另外四份
cache/state及Native/PTO保护区检查通过。B4三个图变体的输出、Top-K及完整allocation
graph/eager逐bit一致。48份源码及冻结启动脚本哈希与任务manifest一致。
汇总见`step46_residual_v1/pool_native_continuous/summary.json`，完整输出坐标分析见
`step46_residual_v1/final_attribution.json`，各任务日志已归档。

本次达到降低指定step46最大绝对差的目标，保留上述两份生产文件中的修正；仍有容差内
浮点差异，不声明Native/PTO输出全部逐bit一致。验证范围是原100步容量的47步eager
及B4同图重放，未将历史100步结果算作本次最新源码的复验，未恢复P3其他场景/P4，
正式权重仍等用户通知，阈值与依赖仓库不变。

## 76. 2026-09-23：step46剩余差异分段，WO-B量化边界

本轮按用户“再看看哪里有差异”继续分析第75节最终捕获，不改生产源码、不扩大验收矩阵。
证据目录为`results/cann90_20260921/step46_remaining_v1/`；输入仍为
`step46_residual_v1/pool_native_continuous/b40/output_boundary.pt`。
CPU脚本`analyze_boundaries.py`生成`boundary_map.json`，保存完整差异坐标及历史写入来源。
表中的差异为逐bit比较，最终输出超出原冻结容差的元素仍为0。

| 边界 | 剩余差异 | 当前定位 |
| --- | --- | --- |
| QR INT8及FP32 scale | 0 | 全240行相同 |
| Query BF16 | 6/7864320 | query3/head15三处，query89/head9三处，均为非RoPE列；max_abs=0.0078125 |
| selected SWA KV | 42次读取、7个独立元素 | 6个历史token被30个query读取；max_abs=0.00390625 |
| selected压缩KV | 0 | 所有有效selected元素相同；完整allocation的47步证据见第75节 |
| attention加逆RoPE，使用实际PTO输入 | 3471/7864320 | 包含Query、SWA及本段计算差异 |
| 同段给定Native Query | 3132 | 仍保留PTO cache |
| 同段给定Native Query和KV | 31 | max_abs=0.000244140625，排除本次Query/cache输入差异 |
| O投影给定Native输入 | 27211/983040 | max_abs=0.00390625，RMSE=0.0001387983211；进一步拆分如下 |

7个独立SWA元素的最后写入来源如下，位置和输入行均从原mixed轨迹计算，尚未对这些写入步
另跑WKV/RMS隔离，不能直接宣称全部由RMS或矩阵乘引起：

| request | KV position | column | 最后写入step/input_row |
| --- | --- | --- | --- |
| 19 | 131181 | 310 | 29/115 |
| 20 | 131177 | 490、491 | 28/122 |
| 22 | 131231 | 110 | 42/133 |
| 34 | 131166 | 66 | 25/204 |
| 34 | 131221 | 286 | 39/207 |
| 39 | 131239 | 266 | 44/236 |

输出的边界替换结果：实际PTO为65485个不同BF16元素，换Native Query为62485个，
再换Native KV为33808个，直接给Native O投影输入为27211个。替换会改变后续量化边界，
这些计数不能相减后当成可相加的独立误差贡献，也不能把Query中间值max_abs当最终输出max_abs。

新增`dsv4_csa_o_projection_boundary.py`，通过现有`diagnose_o_projection`复用完整生产
O投影计算。首先逐bit复现保存的PTO O投影输出，再将Native BF16 WO-A结果填入输入的
相应列，用单位WO-A权重精确透传，保留240行及全部8192列token量化语义。透传后的输出
与原PTO O投影输出逐bit相同，仍有27211个Native差异；这说明该边界的输出差异不因
替换WO-A结果而变化，并不声称已经直接捕获并证明全部WO-A内部FP32累加值相同。

在同一生产函数中使用两组WO-B选择矩阵，分别观察量化结果的前/后4096列。
输出为BF16(q×token_scale)，用已知scale恢复INT8，并对所有观察值验证恢复后重新乘scale
转BF16逐bit相同。给定Native WO-A时，PTO量化与Native有53个INT8元素差异，每个差1。
Native npu_dynamic_quant重新执行所得INT8及scale均逐bit复现保存值。
这些差异接近±63.5舍入边界，例如[1,6180]输入0.51171875、scale=0.008058562874794006，
Native=63，PTO=64；[47,6427]则Native=64，PTO=63。因此不能靠统一改成向零舍入修复。
CPU照公式计算的reciprocal再乘只有50个Native差异，与设备PTO本身仍有9个INT8不同，
说明不能以CPU倒数直接代替A3数值行为。下一步须对齐Native实际设备量化乘数计算路径。

对量化后的有界整数点积用CPU FP64精确累加，再按实际FP32顺序反量化：
使用恢复的PTO量化值及当前先激活scale、后权重scale的顺序，全部983040个输出逐bit
复现设备结果，闭合上述归因。给定Native量化值时，当前顺序还剩8个BF16差异，
改为先权重、后激活scale为0个，合并scale则为11个。这里只是诊断对比，未改生产反量化。
这也再次确认WO-B不能直接沿用Q投影的合并scale顺序。

O投影任务`task_20260923_003249_25198713728`完成exit0，报告为DIAGNOSTIC_COMPLETE；
报告中的逐bitNative对照FAIL表示已测到的边界差异，不是冻结精度验收失败。
证据：`o_projection/o_projection_boundary.json`、对应`.pt`、`quant_coordinates.json`
及`quant_cpu_orders.json`。

另用已有sparse诊断复现query5、给定Native Query和KV，任务
`task_20260923_003250_25206269487`完成exit0，重映射后24行全部逐bit复现保存的PTO结果。
仅[head35,col162]与Native不同：Native=-0.0286865234375、PTO=-0.02880859375。
捕获的PTO numerator=-4.556391716003418、denominator=158.49664306640625；
CPU FP32除法=-0.02874756045639515，FP64除法=-0.028747559745441423，均在BF16
中点-0.02874755859375的PTO一侧。故对这个点，仅替换最终除法不足以恢复Native；
仍需核对上游累加/归约，当前没有Native numerator/denominator捕获，不武断归因具体指令。
证据：`sparse_q5/sparse_boundary.json`、对应`.pt`和`ratio_boundary.json`。

本轮仅执行上述两项小重放，均exit0，源码哈希在任务后核对一致，任务日志已归档。
生产计算、依赖、Native及冻结容差均未修改；最新完整CSA输出仍是第75节的max_abs=0.00390625。
优先处理方向为WO-B量化边界，其次是6个SWA历史token的投影/RMS隔离，再处理Query的6处
及给定相同输入时attention的31处；未将本轮诊断记为新的连续100步或完整P3/P4验收。

## 77. 2026-09-23：修正WO-B量化乘数与反量化顺序

只读CANN A3 dynamic_quant源码确认，量化乘数直接通过设备向量Div计算127/amax；
输出dequant scale独立计算amax×(1/127)。旧PTO先得到dequant scale再recip，数学等价
但中间舍入不同。生产`decode_o_proj.py`在原token scale任务中同时计算并保存这两个值，
量化任务直接使用127/amax，仍保持原RINT→FP16→INT8转换。新增的是单个token尺度的
核内GM scratch，没有新增外部PTO提交或host取值。WO-B反量化按Native改为先乘channel
weight scale，再乘token scale；没有套用Q投影的合并scale顺序。

完整CSA CPU lowering通过。小重放`task_20260923_090422_204211827553`完成exit0，
给定Native WO-A后的1966080个INT8值全部逐bit相同；给定Native O投影输入的983040个
BF16输出也全部逐bit相同，旧版分别有53和27211个差异。测试新增candidate模式严格要求
上述两项逐bitPASS，并继续通过选择矩阵恢复量化值及精确整数点积重建来核对实际设备结果。

原mixed100容量的B40 eager step0～46任务`task_20260923_090518_206026326902`完成exit0，
全部846项通过；B4同图A→B→A任务`task_20260923_090520_20613002997`完成exit0，90项通过。
第46步不同BF16元素由65485降至41635，RMSE由0.0002252867707降至0.0001830331603，
max_abs仍为0.00390625，冻结容差超差为0。新旧采集Native输出逐bit相同；完整链诊断中
给Native O投影输入后的输出也全部一致，与小重放吻合。

证据：`results/cann90_20260921/wo_b_fix_v1/`下`summary.json`、`manifest.json`、
`native_arithmetic_evidence.json`、`local/o_projection_boundary.json`及B40/B4任务日志。
49份源码在任务结束后按冻结manifest核对一致。保留生产WO-B修正，本轮未改依赖、Native
或冻结阈值；未重跑最新源码完整100步，不能将47步诊断写成100步验收。
本节数值仍使用cann_recipe参考checkpoint，不能作为新到位正式权重的精度结论。

## 78. 2026-09-23：正式ModelSlim W8A8权重到位，开始Native加载核对

用户已明确通知`/data/model/DeepSeek-V4-Flash-0731-w8a8`下载完成，原等待通知限制解除。
其索引为`quant_model_weights.safetensors.index.json`，不是参考权重的model.safetensors索引。
遍历索引的75个分片，检查文件存在、safetensors header、所有索引tensor存在以及data_offsets
未超过文件实际长度，全部通过；这是结构/长度检查，不冒充发布方校验和验证。
量化描述为W8A8_DYNAMIC，CSA layer2的24个参数及weight_scale、weight_offset均已保存。
证据：`results/cann90_20260921/formal_weights_20260923/inventory.json`。

测试配置对正式checkpoint显式使用quantization=ascend，沿用Native VllmConfig创建的
AscendModelSlimConfig及其DeepSeek前缀映射。新增独立`dsv4_csa_formal_weights.py`加载路径，
直接读正式索引，严格匹配参数名/shape，使用Native parameter.weight_loader与统一
process_weights_after_loading；记录并核对原始及Native加载dtype，保留正常Native dtype转换。
不进入参考checkpoint的.scale别名、scale维度补齐或缺失offset补零分支。
完整比较测试据此允许正式ModelSlim权重并标明单层scope，不宣称全模型加载或服务接入完成。

Native单层加载和ND布局任务`task_20260923_091048_22344399272`已提交，结果待归档；
冻结源和启动脚本位于上述正式权重证据目录，未预记P0/P2通过。

首轮任务在构造ModelConfig时因独立测试未导入Native量化注册而exit1，尚未加载权重。
补充ModelSlim类注册后，v2任务`task_20260923_091256_242016730267`完成exit0：正式24个
CSA参数全部经Native参数加载器读入并精确核对；Native post-load后全部参数格式为ND=2，
五组Native cache布局检查通过。q_norm/kv_norm及三份量化scale加载为BF16，两个compressor
norm保持FP32，均遵循Native加载dtype，未自行保留checkpoint FP32或改变Native精度。
本结果仅为正式权重单CSA层加载和ND/cache布局通过，不等于完整P0或全模型加载验收。

v2正式权重B4/B40完整CSA比较分别提交为`task_20260923_091351_249727210972`、
`task_20260923_091352_249819028382`，history131071/seed1024；结果待归档。

上述正式B4/B40任务均完成exit0，输出、六份cache/state、Top-K及Native/PTO未写区域
检查全部通过。B4输出max_abs=0.00390625、RMSE=0.0002500174742；B40输出
max_abs=0.00390625、RMSE=0.0001217815443；二者冻结容差超差均为0，Top-K逐bit相同。
权重和Native metadata/cache/state均经现有PTO接口进入完整CSA，没有fallback或替换golden。
50份冻结源码在任务后核对一致，三项v2任务命令、日志及summary.json均已归档。

正式P2当前为18组中的2组（B4/B40、history131071），其余16组以及正式连续轨迹/图重放
尚未执行；不能把参考权重的936项或旧100步结果计作正式权重验收。服务forward派发、P3其余
场景、P4及P5状态不变。本轮生产改动仅为第77节WO-B；新增正式加载支持位于测试入口。

## 79. 2026-09-23：全部切换正式权重，定位SWA RMS归约差异

用户明确要求后续全部使用`/data/model/DeepSeek-V4-Flash-0731-w8a8`。本节所有设备诊断、
完整CSA及图重放均使用正式ModelSlim权重；旧cann_recipe只保留既有历史证据，不再运行。
连续/图重放入口使用第78节已核对的Native正式加载分支，并标明正式单层scope。

修正前正式B40任务`task_20260923_092455_30093173400`完成exit0，保留原mixed100容量
131683，只执行eager step0～46，846项全部通过（报告DIAGNOSTIC_COMPLETE，不是100步验收）。
第46步max_abs=0.00390625、RMSE=0.0001485185494，32707个BF16元素不同，冻结超差为0。
Query、QR INT8/scale、选中压缩KV，以及给定Native输入的O投影均逐bit一致。
SWA选中KV有54个不同读出，实际为6个历史token的9个独立元素，源于step14/18/19/32/34/42。
给定Native Query及KV，attention加逆RoPE有35个BF16元素不同，对应最终输出5279个不同。
证据：`formal_step46_v1/b40/output_boundary.{json,pt}`、`swa_origins.json`；运行后冻结哈希核对一致。

新增`dsv4_csa_swa_rms_boundary.py`，保留原始M240投影尺寸，直接复用生产
`kv_project_native_240`及`kv_proj_rope`，与Native F.linear、npu_rms_norm比较。
诊断前两次提交因内联task依赖未绑定到局部变量而lowering失败；第三次因诊断Tensor表达式
使用普通算术而解析失败。修正仅在本仓库诊断脚本，未修改PyPTO或依赖，失败证据保留在
`formal_swa_rms_v1/v2/v3`，均不计数值PASS。

第四次`task_20260923_093332_235200114879`完成exit0。上述6个源步骤加step46共1680行，
生产WKV的BF16投影与Native全部逐bit一致；PTO RMS输出分别有1/1/1/1/1/4/0个不同元素。
给PTO投影后调用Native RMS，输出全部一致，定位到RMS而非矩阵乘。
Native CANN9.0 `rms_norm/reduce_common.h::ReduceSumMultiN`实际路径先从零开始按列累加
8组64列，再WholeReduceSum；旧PTO先对各64列归约再累加，两者FP32舍入顺序不同。
按Native逐列累加，再Sqrt及向量Div(1,root)，1680个rstd全部逐bit一致，BF16归一化全部一致。
折半归约虽在这些样本也得到一致BF16，仍有20个rstd末位差，故未选用折半顺序。
CPU标量sqrt/reciprocal也有末位差，不能套用QR的标量修正到这一Native向量RMS路径。
证据：`formal_swa_rms_v4/swa_rms_boundary.{json,pt}`及冻结诊断脚本。

生产修改仅在`qkv_proj_rope.py`的SWA完整块/尾块：改为64列向量逐列累加后归约，
并显式Sqrt+Div，与Native路径一致。没有新增外部适配、host取值或依赖修改。
完整CSA CPU lowering通过。正式B40原容量47步及正式B4同图A→B→A已提交，待结果归档；
另采正式query9的attention累加器，继续定位相同输入下的剩余舍入边界。

修正后正式B40 `task_20260923_093600_404821116244`完成exit0，eager0～46共846项通过；
六份cache/state的每一步max_abs均为0，逐元素数值相同，Top-K和保护区通过。
第46步选中SWA差异54→0，最终输出不同元素32707→5279，RMSE从0.0001485185494降至
0.00005738172331，max_abs仍为0.00390625，冻结超差为0。Native输出前后完全相同；
修正后完整PTO输出与修正前“给定Native Query/Native cache”的重放输出完全相同，
闭合SWA归因。Query、QR/scale与给定Native输入的O投影仍全部相同。
正式B4 `task_20260923_093600_4047695745`同图A→B→A完成exit0，90项通过；
地址固定、真实Native ExternalEvent，graph/eager输出及整份allocation逐bit相同。
此次总计936项通过，51份冻结源码运行后全部核对一致；没有重跑正式100步或完整P3。
证据：`formal_swa_rms_fix_v1/summary.json`、`manifest.json`、`production.patch`及任务日志。

剩余35个attention加逆RoPE的BF16差异在修正前后完全相同，已排除SWA、Query、QR和O投影。
正式query9小重放`task_20260923_093635_3296115225`完成exit0，24个重复行均逐bit复现
完整链保存的PTO结果，仅head38/col111不同：Native=0.02392578125、PTO=0.0240478515625。
PTO分子2.890251874923706、分母120.49333953857422；FP32商0.02398681826889515、
FP64商0.0239868185743033，均在BF16中点0.02398681640625的PTO一侧。
Native A3 RescaleO源码同样使用向量Div和BF16 CAST_RINT；没有证据支持改最终舍入模式。
因此单纯提高最终除法精度不能修复此点，下一步核对attention上游累加与归约。
未采到Native内部累加器，不把具体来源预判为QK、softmax或PV中的某一段。
证据：`formal_swa_rms_fix_v1/sparse_q9/sparse_boundary.{json,pt}`及`ratio_boundary.json`。


## 80. 2026-09-23：算子修正独立提交并push，后续精度暂停

用户要求暂停其他精度工作，先push当前算子修正，再列出后续可做事项。
提交`18ec20a`（fix(pto): align DSV4 CSA arithmetic with Native on A3），已push到
`origin/dsv4-flash-pto`；本次仅一个commit，包含qkv_proj_rope、decode_compressor_ratio4、
decode_sparse_attn_csa、decode_o_proj四份生产算子文件，共185行新增、113行删除。
提交说明详细记录修正原因、正式权重936项既有验证、残留attention差异及验收范围。
四份提交源码与第79节冻结验证源码一致。按用户此前要求不运行提交检查或追加设备测试；
测试、文档、产物及其他未提交修改均留在工作区，本次未推送。
后续精度排查暂停；此前暂停的P3其他场景和P4也未自动恢复。


## 81. 2026-09-23：真实服务 forward 入口与正式权重单层验证

用户要求接入真实服务 forward，继续暂停剩余精度排查。新增显式模型架构
`PyptoCSADeepseekV4ForCausalLM`，继承 Native 模型/权重加载，只替换 target C4 attention
实例。真实调用经 `attention.forward -> dsv4_csa_forward -> CSAServiceRuntime -> 一次完整CSA`。
Native post-load hook 后准备权重和工作区，普通 warmup 准备 Native Hadamard；positions、
metadata、cache/state 均从本次 Native context 绑定。保留 Native 三类 metadata 事件等待和
connector 的 wait/notify/save。无 metadata 的 profiling 及不满足接入条件的调用走 Native。

runner 增加仅针对显式 CSA 架构的 graph 选择限制，防止 padded 请求误用已捕获的 PTO 图。
metadata 新增 host max_query_len，结合 num_actual_tokens/num_decodes 证明无 padding 的均匀
S6，不从 device 取 query 长度。没有改动 DP padding 协议；同步 padding 的 DP full graph
暂不启用，P3/P4原暂停项不自动计入通过。详细入口见 `DSV4_FLASH_CSA_SERVICE_FORWARD.md`。

首次 task_20260923_105002_344865722443 因测试脚本误导入 `tolerances` 失败，算子未运行；
改为已有 `tolerance`，并修正小 batch 的 BatchDescriptor 和 ExternalEvent 报告时机。
B4 task_20260923_111012_87584012717、B40 task_20260923_111128_11960056013 均 exit0。
正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，history131071/131073、页表交换、
同图 A→B→A 各87项检查通过；输出使用冻结容差，Top-K精确相同、六份cache/state及保护区通过，
graph/eager 输出和完整allocation逐bit相同。B4 variant B 输出max_abs=0.0078125，
其余B4及B40为0.00390625，冻结超差均0；不声称与Native逐bit一致，不继续追查末位差异。
实际 torch.compile(backend=eager, fullgraph=True) 边界通过；profiling 和当时未启用的B1
Native回退的输出、cache逐bit一致。此处的编译测试不是完整vLLM后端编译或HTTP服务验收。

B4启动占位输入（seq_len=6、position=127、空页表、负slot）warmup/capture，随后恢复真实
请求并重放，task_20260923_111521_2203692143 exit0，另87项通过。不是完整P3 dummy覆盖。
CPU真实runner分派及host门禁首轮13项通过；EngineArgs实际配置解析选中新增架构，drafter
仍为DSparkDraftModel，FULL_DECODE_ONLY、warmup=1、MRv1。21个目标C4层的量化描述与
layer2一致；只执行过layer2，不能当成21层加载/运行通过。

产物：`service_forward_v2/{b4,b40}/service_forward.json`、`cpu.xml`、`engine_config.json`；
`service_forward_dummy_v1/b4/service_forward.json`及源码hash。
发现本机CANN扩展此前仅通过测试helper导入；为普通服务进程增加两份本地忽略跟踪的.so链接，
指向已构建的 `.cache/csa/native-install/`，未改二进制或依赖仓库。
本次没有commit/push；本节证据对应删除离散batch白名单之前，后续版本见第82节。

## 82. 2026-09-23：按用户要求改为动态 batch，修复 Indexer 尾块

用户指出 batch 不应受 B4/8/16/24/32/40 测试枚举限制。核对算子已有B_DYN/T_DYN，
问题在适配层把测试矩阵误用为支持白名单。现移除白名单，实际B从Native Tensor形状读取，
服务workspace按min(max_num_seqs,64)分配，接受容量内任意正B且T=B*6；64来自现有算子
内部workspace容量。graph仍按bucket捕获，PTO bucket必须精确匹配实际S6请求，不能把
动态算子等同于同一张ACL Graph可任意改变形状。S=6和TP1等现有约束保持。

放开B1后 task_20260923_111935_321482329326 在首次warmup失败，设备日志明确断言
`block_num >= 1`。定位到Indexer `dq_rope_units=(T//8)*16`：T=6得到0，其他非8整除的T
还会漏掉尾部。修改为ceil(T/8)，给部分tile有效行数的首版又在B1/B3触发AIV异常
（task_20260923_112341_406320120141 / task_20260923_112341_406119713546）；代码复核发现
广播输入的有效行数与8行目标不一致。现保留原8行完整块，最后不足8行按实际单行执行
相同反量化/RoPE；不引入host补齐、额外算子或依赖修改，不改已有精度算法。
下一版因分支复用不同shape临时变量名称而lowering失败，改用独立tail变量后完整CSA CPU
lowering通过。CPU服务门禁与实际runner分派当前17项通过。

当前最新源码冻结在 `service_forward_dynamic_v3/source/`，含manifest；B1/B3启动占位capture
及正式A→B→A任务：task_20260923_112741_7110028905、task_20260923_112741_7026411360。
同一服务层/算子在B1→2→3→5→40→1间切换并检查编译artifact复用的任务：
task_20260923_113052_16988418065。11:30共享机16张卡由其他任务占用，上述三项排队，
尚无最新动态版本的真机PASS；不得以旧B4/B40通过替代这轮结果。


第82节续：排队结束后v3三项均在PTOAS阶段失败，原因是尾块1×1 FP32 Tensor不满足
32字节行对齐（不是设备执行PASS）。尾块scale改为设备端 `pl.read` 标量，再乘1×128
权重scale；没有host取值。完整CPU lowering+PTOAS代码生成随后通过，见
`service_forward_dynamic_v4/cpu_codegen_v2.log`。最新冻结源码为v4，三项重提：
B1 task_20260923_113741_245921129591，B3 task_20260923_113741_24592783972，
同进程batch切换 task_20260923_113741_245459626332，提交时待结果。
普通环境不经过activate/load_native_extension的Native扩展和模型类导入也已通过。

第82节续：v4三项CPU编译后仍在首次设备执行出现AIV异常，尚无数值结果。
进一步检查发现既有QKV RoPE尾块也存在有效行数不一致：sin只有实际行，sign仍为8行；
Q的RoPE乘法也把8行数据与不足8行的cos/sin相乘。此前B4/8/16/24/32/40对应T均为8的
整数倍，未进入这些分支。v5只在尾块将sign和Q RoPE操作数的有效行数设置为实际行数，
保持满块及数值算法不变。完整CPU lowering+PTOAS通过，源码冻结于
`service_forward_dynamic_v5/source/`。B1启动占位capture及A→B→A验证任务
`task_20260923_114612_401642117676`已提交，设备结果待确认；不修改依赖仓库。

v5 B1任务exit1，仍在首次warmup出现AIV异常，没有数值结果。上面的有效行数修正是源码
检查所得，不能单独解释此次设备异常。停止重复完整矩阵，提交正式权重B1的QKV、Indexer
独立执行诊断：task_20260923_115525_149691023830（QKV）；结果见v5下对应目录。
这些诊断只定位执行异常，不恢复此前暂停的精度逐bit排查。


## 83. 2026-09-23：动态 batch 完整链路尾块修复，正式 B1 与同产物切换通过

延续用户要求：配置容量与运行时实际B分开，不再用测试枚举限制B。上一节QKV独立执行
`task_20260923_115525_149691023830`通过；Indexer首次诊断误将Tensor当tuple取第0行，
修正测试取值后 `task_20260923_115627_166151832603`通过。二者只用于定位执行异常，
不是单独的数值验收。带运行时日志的正式B1仍失败（`task_20260923_115905_274708222025`），
因此不能把此前QKV有效行数问题单独认定为完整链路异常的根因。

继续检查完整链路发现真实越界/漏行：attention规划固定遍历8行，在T=6时仍读第6/7行
position、Top-K与页表；普通RoPE符号处理固定4行且末块未裁剪；KV写回和逆RoPE按T//8
计块会漏尾；O投影最终写回固定8行会越界，量化清零从未对齐T开始还会超出最后pad边界。
修正这些位置的ceil分块、实际行循环、显式valid_shape及store边界；保留固定内部tile，
不增加host padding、device窗口搬运、外部适配调用或任何依赖仓库修改。
显式Tile版row_max提供同尺寸临时tile；Tensor级输出用assemble携带有效行元数据。
CPU完整lowering和PTOAS生成通过，见 `service_forward_dynamic_v6/cpu_codegen_v4.log`。

最终冻结源码 `service_forward_dynamic_v6/source/` 与当前生产/测试文件hash一致。
正式权重始终为 `/data/model/DeepSeek-V4-Flash-0731-w8a8`。两项任务均exit0：

- B1启动占位warmup/capture、真实history131071/131073、页表交换、同图A→B→A：
  `task_20260923_120750_225510828950`，87项通过。Native ExternalEvent、真实安装后的
  attention.forward、torch.compile边界、profiling/非均匀边界Native回退均通过。
  graph/eager输出及完整allocation逐bit相同；Native输出max_abs=0.00390625，冻结超差0。
- 同一进程/服务层/已注册CSA算子按B1→2→3→4→5→40→1切换：
  `task_20260923_120750_225124330269`，7×13=91项通过。每步恰好一次PTO提交；
  JIT specialization始终只有一个，artifact对象不变，证明实际B变化没有重新编译算子。
  输出max_abs均0.00390625、冻结超差0；Top-K精确相同，六份cache/state及未写区域通过。

证据为 `service_forward_dynamic_v6/b1/service_forward.json` 与
`service_forward_dynamic_v6/batch_switch/service_dynamic.json`。本轮合计178项主检查，
不将通过解释为Native输出逐bit一致、全B范围逐项验收、连续精度或完整P3/P4/P5通过。
服务容量为min(max_num_seqs,64)，其中64来自现有内部workspace上限；实际B动态，S仍为6。
ACL Graph仍按各自形状bucket捕获，不能将算子动态B等同于同一张图任意改变形状。
用户暂停的其他精度排查和P3/P4未恢复；此次未commit/push。

## 84. 2026-09-23：P TP4×DP4 离线缓存 → D TP1×DP/EP16

用户确定不依赖同时在线的P/D服务，先由P生成多场景离线KV cache，再释放16卡供D使用。
新增测试专用 `offline_pd/connector.py`、`offline_pd/run.py` 和
`DSV4_FLASH_CSA_OFFLINE_PD.md`。缓存范围包含所有target与draft组，按Native各组
逻辑页号落盘/恢复，包含SWA、C4/C128、Indexer及compressor state；P计算H、D提交H+1，
复用Native N−1边界。Native/PTO两轮D使用同一bank，TP1×DP16且开启EP16。

计划场景H=255/4095/32767/131071/131072/131073，每长度4份确定性token序列，
每个P DP rank一份。D每卡BS扫描1/4/8/16/24/32/40，首轮仅均衡负载、eager。
不恢复暂停的其他P3/P4和精度排查；离线方式不等于在线Mooncake或完整P5验收。

CPU检查通过connector非抽象类及SWA页号保持/空洞过滤。
短场景任务 `task_20260923_135331_215897732379` 在LLM构造前失败：
Python入口尚未执行Native CLI的pre_register，int8 indexer被上游Literal拒绝。
入口补上既有 `current_platform.pre_register_and_update()`，未改生产依赖。
重提 `task_20260923_135431_21771739041` 进入16worker启动，但ATB注册缺少
`libatb.so`，任务失败，无缓存产出。本机存在NNAL9.0.0安装包，正在恢复本地ATB环境；
尚无P完整prefill、合格缓存、D16接入或性能PASS。

正式场景plan：`offline_pd_bank_v1/plan.json`；短场景：
`offline_pd_smoke_v1/bank/plan.json`，对应启动日志在同目录prefill/prefill_v2。
当前输出的elapsed_including_io_seconds包含IO，只用于过程记录，不计decode性能。

ATB补充：本机NNAL9.0.0普通安装遇到系统CANN所有者检查，改用安装包原生
`--noexec --extract`在工作区提取ATB运行库，以原始set_env配置CXX11 ABI1；
未修改系统CANN、torch_npu或组件仓库。CPU ATB注册通过，`../env.sh`已接入。
重提P任务 `task_20260923_140032_228587613477`，日志确认TP4×DP4、EP16，
16个rank通信初始化完成，正在加载正式75分片；结果仍PENDING。


## 85. 2026-09-23：官方 v0.25.1rc1 基线迁移

按用户要求从官方 `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` 创建
`dsv4-flash-pto-v0.25.1rc1`，新工作目录 `../vllm-ascend-dsv4-pto-0251rc1`。
旧分支、旧工作目录及 WIP 完整保留。迁移已有独立 CSA、精度修正、动态 batch 和服务入口，
接入 release `.decode` metadata、Native compact producer 及加载后 runner hook。
原 main QLIv2 / scatter_nd_update_sk 修复不适用于该版本的对应代码，原补丁已归档。
配套 vLLM 固定为官方 v0.25.1（752a3a504485790a2e8491cacbb35c137339ad34），
新 Python 环境为 `../.venv-dsv4-0251rc1`，不替换原 `.venv`。

本轮 CPU：真实模型 / metadata / custom op / runner 导入通过；服务选择、图派发与
release metadata 零拷贝绑定共18项通过；完整 CSA a2a3 lowering 通过。
未运行提交检查或完整精度矩阵。未重建 release Native 扩展 / 算子包，未进行新基线 NPU 验证。

第84节补记：P任务 task_20260923_140032_228587613477 最终因 aclnnHcPre 缺失失败，
正式75分片及 draft 已加载，未产出离线缓存。旧15算子包补建在切换基线时停止，未安装。

用户要求过程文件一并迁移：旧 tests/pypto_test 全目录已复制，原接口入口另存档；
Git纳入文档、脚本、日志、报告及profiler记录。305份 .pt 大快照保留本地原路径，
SHA256与文件大小见 handoff/MIGRATION_PROCESS_FILES.json。
证据与接续步骤见 [基线迁移说明](BASELINE_MIGRATION_V0251RC1.md)，
新CPU证据在 results/migration_v0.25.1rc1_20260923/。

## 86. 2026-09-23：删除旧分支/工作目录，恢复离线 P/D 工作

用户要求继续迁移前的 P TP4×DP4 → D TP1×DP/EP16，并明确允许删除旧目录及库上旧分支。
先核对305份本地大快照均存在且大小符合已保存SHA256清单，迁出PTOAS、ATB和构建工具到
工作区 `.cache/dsv4-toolchain`，保存最终旧WIP、入口及环境快照；共享Git目录迁至
`.git-repositories/vllm-ascend.git`，修复新目录和两个独立bugfix worktree引用。
新分支e048502推送并核对远端SHA后，以expected-old-SHA lease删除远端dsv4-flash-pto，
再删除其本地分支和旧工作目录。Native安装包只读目录恢复当前用户删除权限后清理完成。
未删除nalinaly/vllm-ascend仓库及其他远端分支。

新基线C++ Native扩展以CANN9.0.0、torch2.10.0、系统GCC10、CMake3.31、Make -j8构建，
安装到新目录 `.cache/csa/native-install`，真实CPU导入通过。源码未加入旧main的CANN兼容补丁。
正在从release自带csrc源码构建14算子custom包，包含HcPre/HcPost、CSA所需Native算子和MoE算子；
编译日志在 `.cache/csa/setup-release/`。构建完成、符号检查和真机执行前不记HcPre已可用。

离线脚本按release去掉不支持的indexer_kv_dtype=int8配置，实际A3 Indexer仍为INT8。
该vLLM没有main的kv_connector_block_state，改保留update_state_after_alloc返回的KVCacheBlocks
中Native manager持有的各组list引用，在最终prefill chunk读取最新块号，保留SWA空洞与后续追加；
未修改vLLM调度器。CPU检查通过真实KVCacheBlocks的替换/追加可见性及C4/C128压缩页数。
压缩缓存只保存floor(H/ratio)个有效压缩行覆盖的页，避免把speculation预分配页当作前缀缓存。
P计划使用正式权重、history255×4先走通；尚未生成release离线缓存。

本次14算子包源码均为官方v0.25.1rc1已有csrc；`git diff 9bf964cb4b87c8cd0d6852c41a55b3c29711fa95 -- csrc`
为空。attention目录包含compressor、compressor_metadata、vllm_quant_lightning_indexer、
vllm_quant_lightning_indexer_metadata、sparse_attn_sharedkv、sparse_attn_sharedkv_metadata、
rms_norm_dynamic_quant、inplace_partial_rotary_mul；moe目录包含scatter_nd_update_v2、hc_pre、
hc_post、moe_gating_top_k_hash、moe_gating_top_k、dequant_swiglu_quant。
HcPre/HcPost设备入口分别为`csrc/moe/hc_pre/op_kernel/hc_pre.cpp`和
`csrc/moe/hc_post/op_kernel/hc_post.cpp`，定义及tiling在各自op_host目录。
`csrc/torch_binding.cpp`中npu_hc_pre_v2走run_hc_pre_fusion→aclnnHcPre，npu_hc_post走aclnnHcPost。
这是Native服务依赖的恢复，不是新增14个PTO算子；独立PTO CSA的范围未改变。

14算子包于15:06构建完成(exit 0)，安装到本地`.cache/csa/csa-native-ops-install`。
`libcust_opapi.so`已导出aclnnHcPre/HcPost及CSA所需API，env.sh现加载该release包。
源码树、包SHA256、API符号及构建/安装日志保存在`results/release_offline_pd_20260923/native_build/`。
真机执行任务`task_20260923_150737_24250853714`已提交：正式layer2 HC权重及三种Native metadata；结果待定。

任务`task_20260923_150737_24250853714`完成(exit 0)：正式checkpoint layer2的HcPre/HcPost
在A3实际执行PASS；SAS、QLI、Compressor三种metadata执行均PASS。该结果只确认算子可用，
不代表完整模型或精度验证。HcPre缺失问题已消除。P TP4×DP4/EP16短场景任务
`task_20260923_150829_249108112866`已提交，输入为release_offline_pd_20260923/smoke_bank。

15:18:30检查：P任务task_20260923_150829_249108112866自15:08:29提交后累计排队10分钟，
仍为pending；当前其他任务占用16张卡。本次主动等待按10分钟上限结束，保留原排队任务，
没有重复提交或取消。7200秒为启动后运行超时，不是排队超时；尚无P缓存产出。

15:29检查：P任务task_20260923_150829_249108112866已转为running，约15:28获得全部16卡；
四个P DP进程启动，日志进入world_size=16的HCCL初始化。尚无缓存产出，继续核对实际执行。


## 87. 2026-09-23：release P缓存产出、前缀边界修正、提交D16

P任务`task_20260923_150829_249108112866`约15:28获得16卡，完成正式模型与draft加载、
启动预热和H255×4的prefill，exit0并释放全部卡。每rank模型权重22.8821GB；四个DP场景
各落盘四个TP副本，共16份cache.safetensors，每份191个tensor，总约354MiB。
路径为`results/release_offline_pd_20260923/smoke_bank/`。大payload留本地并加入ignore，
路径和大小在audit.json中，日志在prefill_v1/rank*.log。没有继续重复P或重跑精度矩阵。

首次audit FAIL：脚本未识别release的mtp.0/1/2名字；另mtp.1/mtp.2原始副本末页第31行不同，
即全局position255。H255的历史有效范围是0～254，DSpark已把未来预测写入position255，
该行不属于本次P→D应恢复的前缀。43个target层与mtp.0原始缓存相同。
原始payload和首次失败audit_raw_v1.json完整保留；不修改Native或PTO计算，也不恢复逐bit精度排查。

新增测试专用prefix.py：CPU保存/恢复副本清除H之后的行，压缩缓存边界为floor(H/ratio)，
已有schema1 bank在D恢复时使用相同边界；不改有效行、不改Native allocation或算子热路径。
层覆盖由正式config明确生成43个target层和3个mtp层的191个tensor合同。
单次CPU边界检查覆盖ratio1/4/128：原始输入不变、有效行逐bit保留、最后有效行差异不会被屏蔽。

用户明确要求“不要搞什么hash校验”：已从offline_pd的plan/audit/save/load路径移除hash生成
与校验，包括模型配置、token和缓存payload；不再将hash作为任何D启动前提。
改为直接比较prompt token列表、必要layout及有效前缀tensor字节。
最终audit PASS：四场景、四TP副本、全部191个tensor的有效前缀逐bit一致。
早期产物中已有的摘要字段仅作为原始历史记录保留，新流程不读取/计算它们。

D任务`task_20260923_154135_38813947929`已提交，TP1×DP16/EP16，每卡B4，先Native再PTO，
两者使用同一H255×4 bank。输出目录为decode_native_b4_v1和decode_pto_b4_v1。
尚无D运行结果；该轮先验证接入，包含IO的elapsed不作为稳态性能结果。

D任务约16:09获得16卡。Native轮完成16rank×4request，64次OFFLINE_CACHE_LOADED，
各请求均输出128token，21个target CSA层逐rank均有执行记录。16份rank报告齐全，
Native执行汇总见decode_native_b4_v1/summary.json；这不是输出精度或稳态性能验收。
PTO轮已自动启动，结果待定。


## 88. 2026-09-23：Native D16通过，PTO初始化norm dtype修正

> 本节的初始化FP32转换方案已被用户否决；未上卡，排队任务已取消。实际修正见第89节。

任务`task_20260923_154135_38813947929`的Native阶段完成，16份rank报告均有4个请求，
每请求128token，共8192token；64次缓存加载、每rank全部21个target C4层执行。
证据为`release_offline_pd_20260923/decode_native_b4_v1/summary.json`及rank日志/JSON。
这确认短场景P TP4×DP4缓存能在D TP1×DP16/EP16实际恢复并decode，不代表精度或性能验收。

PTO阶段在16:16初始化失败，组合任务最终exit1。root cause为
native_adapter.prepare_weights要求cmp_norm_w FP32，而release A3 Native加载的是BF16。
这是迁移遗漏：release Compressor构造器仅A5指定FP32；A3沿用模型BF16，Native设备kernel
在RMS前把norm转为FP32。该错误在第一份norm检查处中止，尚无PTO CSA实际执行或缓存加载。

只修本仓库native_adapter：两份compressor norm允许已加载BF16/FP32，在模型初始化时
一次性转FP32供现有PTO ABI使用。BF16→FP32保持数值精确，不修改Native parameter或dtype，
不新增forward中的适配、同步或host取值，不改PyPTO/Simpler/pypto-lib/CANN算子。
CPU用正式checkpoint的两份norm值、其他权重meta tensor验证prepare_weights通过，
BF16/FP32来源均精确转换，Native参数对象保持不变；证据pto_norm_prepare_cpu.json。
未重复Native D、未运行提交检查或hash校验。

只重提PTO D B4任务`task_20260923_161904_380071921124`，输出decode_pto_b4_v2；结果待定。
后续24个长短P场景的输入已生成于full_bank/plan.json，尚未执行P长场景。


## 89. 2026-09-23：CSA直接接收Native BF16 norm

用户明确要求Native使用BF16时必须修改PTO算子入口，不接受初始化时转换FP32。
已在排队阶段取消旧方案任务`task_20260923_161904_380071921124`，未执行该方案。
撤销prepare_weights中的BF16/FP32宽松检查和两次.float()，严格接收BF16；
cmp_norm_w[512]和inner_norm_w[128]直接绑定Native原始连续权重。

完整CSA入口以及两路compressor的所有norm参数声明改为BF16。
已有rmsnorm_rope_cache_write和rmsnorm_rope任务加载BF16 gamma tile后cast FP32，
再沿用原RMS/乘gamma/RoPE计算顺序，与release A3 Native的加载及计算dtype一致。
未增加适配调用、GM FP32 norm缓冲、host取值或单独的cast kernel；未修改依赖仓库。

新增一项CPU回归检查：两份norm dtype、数值、data_ptr均保持Native原样，
Native Parameter对象不变，并拒绝向BF16 ABI传入FP32参数；通过。
证据`release_offline_pd_20260923/pto_bf16_norm_cpu.log`。
完整CSA CPU lowering通过，证据`pto_bf16_norm_lower/report.json`。
继续PTOAS代码生成及PTO D16实际接入验证，不重复Native轮或精度矩阵，不执行hash校验。

完整CSA CPU编译含PTOAS通过：`pto_bf16_norm_codegen/report.json`及
`pto_bf16_norm_codegen_v2.log`。生成的两份RMS C++均从BF16 GlobalTensor加载，
并在原有kernel内执行TCVT到FP32。首次CPU编译命令误用RunConfig.output_dir，
在编译前报参数错误；改用本地API的save_kernels_dir后通过，未修改编译器。

PTO D TP1×DP16/EP16 B4已重提为`task_20260923_163126_45104725192`，
使用同一正式权重及smoke_bank，输出`decode_pto_b4_bf16_v3`。
提交时等待16卡资源，尚无设备执行结果，不预记PASS。

16:46状态复查：BF16入口任务task_20260923_163126_45104725192仍pending，
排队约14分钟，16卡被其他任务占用，尚无decode_pto_b4_bf16_v3输出目录。
同时核对正式权重：config含dspark_block_size=5、target_layer_ids=[40,41,42]、
markov_rank=256；quant_model_weights.safetensors.index.json中包含mtp.0/1/2，
共7028个draft相关tensor条目（含量化参数）。Native D rank0日志明确从同一路径
加载DSpark draft，报告loaded:124 params（运行时融合/分片后的参数计数，非checkpoint tensor数），
并出现真实speculative acceptance统计。该目录已包含DSpark权重，无需另配draft目录。

16:50复查：task_20260923_163126_45104725192仍pending，累计排队约19分钟。
16张卡全部由其他任务占用，本任务输出目录仍未创建，尚未启动，无新增执行结果。
保留原任务排队，未重复提交或改动其他用户任务。


## 90. 2026-09-23：整理本次提交范围

按用户要求整理为一次本地提交。tests之外仅4个生产文件：CSA入口、主compressor、
Indexer compressor和Native adapter中的BF16 norm接口与设备tile转换；根目录交接文档
不加入本次更新，当前环境/验证进展统一记入tests下已有迁移说明和本日志。

tests提交release离线P/D接口适配、前缀边界处理、HcPre/HcPost与metadata smoke、
BF16零拷贝CPU回归，以及已完成的P/Native D记录、PTO初始化失败和CPU编译记录。
初始化转FP32的中间方案保留历史证据，明确已废弃，不作为当前实现或设备PASS。
BF16修正后的PTO D16任务仍在等待资源，本次提交不宣称PTO D16精度或性能通过。

大权重不纳入仓库；16份cache.safetensors、未执行长场景的可再生成token列表、
重复编译中间产物留本地。路径及字节数见本轮results/LOCAL_ARTIFACTS.json，
不新增或执行hash校验。只保留两份RMS生成C++作为BF16加载/内部TCVT证据，
加上完整lowering、编译报告及文本日志；不提交.so/.o/.bin/.run等二进制或安装包。
复用已有验证结果，不重跑测试或提交检查；提交说明记录已通过项、失败/取消任务及待验证项。

17:02复查：BF16入口PTO D16任务task_20260923_163126_45104725192仍pending，
累计排队约31分钟。当前14张卡由其他任务占用，仅8、15号卡空闲；本任务需要同时16卡。
队列中另有3个更早提交的16卡任务。本任务输出目录尚未创建，无新增真机执行结果。


## 91. 2026-09-23：BF16准备通过，真实D16暴露缓存共享存储约束

任务`task_20260923_163126_45104725192`约17:19开始，17:22:44在首个PTO调用前失败，exit1。
全部16个rank均完成21个CSA层的BF16权重准备和4个请求缓存恢复，共64次加载。
首次PTO调用在PyPTO的参数描述符别名检查处报错：
`Parameter 'idx_kv_cache' partially overlaps another tensor with a writable alias`。
未进入CSA设备计算，不属于输出精度失败。证据为`decode_pto_b4_bf16_v3/summary.json`及rank日志。

只读检查发现：Native缓存规划允许不同缓存组共享底层分配，依靠独立页表使用不同物理页；
当前PyPTO `_validate_aliases` 仅合并地址、字节数、形状、stride、dtype全部相同的精确视图。
同一字节范围的FP32状态视图与INT8索引缓存视图会触发当前拒绝条件。
新增纯CPU复现`dsv4_csa_shared_storage_repro.py`，调用现有生产视图函数及原始别名检查，
确认上述行为；证据`shared_storage_cpu.json`，不依赖模型数值或NPU执行。

测试工具增加`--layout-only`：加载真实D模型后仅通过RPC记录21个CSA层的缓存描述符、
地址范围及重叠关系，不读取设备张量内容、不启动PTO计算。任务
`task_20260923_172807_209419913258`已运行，输出`decode_cache_layout_v1`，
用于确认实际服务是否属于该共享场景。当前未放宽检查、未修改依赖仓库或生产算子，
也未引入缓存复制来绕开错误。

描述符采集任务已完成exit0：16rank×21层共336份层记录，672对重叠均为完全相同
的起始地址和字节范围，没有真正的部分范围重叠。包括主compress_state与cmp_kv
（FP32/BF16），以及inner_compress_state与idx_kv_cache（FP32/INT8）。
证据decode_cache_layout_v1/summary.json及16份rank*.cache_layout.json。

用户提供PyPTO PR https://github.com/hw-native-sys/pypto/pull/2867 。核对其差异：
_validate_aliases的等价键从地址/字节数/shape/stride/dtype改为地址/字节数，
同时继续拒绝真实的可写部分重叠，正好覆盖本次全部重叠描述符。
本地PyPTO仍为02c0026，包含旧检查，尚未更新依赖或重新进行PTO D16计算。

## 92. 2026-09-23：更新两套调试分支并重测共享缓存接入

按用户要求，两仓库均保持`feat/kernel-mode-integration-test`分支并fast-forward：
PyPTO `02c0026 → 5495749`，包含PR #2867；Simpler `e914837d → 166852bf`。
更新前已有的PyPTO本地torch_npu 2.10.0.post2支持及对应测试、说明继续保留；
完整原始差异另存工作区`.cache/update-20260923/pypto-before.patch`及Git stash。
上游PyPTO的Simpler绑定仍为32dff953，故将原有本地版本绑定和runtime子模块一起
同步到实际安装的166852bf，保持Python ABI、torch_npu扩展和Simpler SDK一致。
没有自行改写别名检查、Simpler运行时实现或CSA生产代码。

在`.venv-dsv4-0251rc1`中从本地源码重新安装，两者使用现有CANN 9.0和GCC 15；
Simpler编译A2/A3 runtime及绑定，PyPTO启用已有torch_npu adapter构建选项。
安装日志保存在工作区`.cache/update-20260923/`，不复制其他checkout的动态库。

扩展CPU复现脚本，增加修复后预期及真实D16描述符回归。更新后的检查接受
同字节范围的FP32/INT8视图，仍拒绝真正的可写部分重叠；16rank×21层全部通过。
证据`shared_storage_updated_cpu.json`及对应日志。这仅验证参数检查，
尚不代表PTO设备执行、输出精度或性能通过；安装完成后重提同一正式权重、
smoke_bank及B4的PTO D16，不重复Native D轮和精度矩阵。

两套安装均完成；安装后的Python ABI、PyPTO torch_npu adapter、Simpler绑定以及
runtime子模块均报告166852bf，torch 2.10.0/torch_npu 2.10.0.post2版本检查通过，
证据`dependencies_updated.json`。PyPTO首次沿用2路构建，调整为显式8路上限时中断
重启；中断轮进入安装阶段后因旧动态库RPATH报错，没有成功安装。后续增量完成
全部编译及链接后重新安装成功，最终日志在`dependency_update/`，不使用中断轮产物。
上游4项别名CPU测试通过；更新后完整CSA CPU编译含PTOAS通过，
证据`dependency_update/alias-ut.log`、`updated_codegen/report.json`。

17:52重提PTO D16 B4任务`task_20260923_175252_286523232409`，输出
`decode_pto_b4_updated_v4`，正式权重及smoke_bank不变，128个生成token/请求。
提交时8张卡被其他任务占用，当前等待16卡资源，尚未启动CSA设备计算。
等待期间仅补一个上游eager真机用例（eager-1），任务
`task_20260923_175431_29119437928`，用于检查新运行时实际下发，不属于模型精度测试。

该eager真机任务完成exit0，1项通过（16.24秒），包含设备标量累加、constexpr
特化及结果检查。证据`updated_eager.xml`、`dependency_update/eager-device.log`。
17:55复查D16任务仍pending；本轮可确认更新、安装、ABI、共享缓存参数检查、
完整CSA编译和基础eager执行通过，不能提前记录D16运行或CSA精度/性能通过。

## 93. 2026-09-23：更新依赖后真实PTO D16首次完整运行通过

复查任务`task_20260923_175252_286523232409`已完成exit0，rank日志显示18:04:22
正常结束。使用正式ModelSlim W8A8权重、既有P TP4×DP4生成的history255离线bank，
D为TP1×DP16/EP16，每rank batch4，64个请求均生成128 token，共8192 token。
全部16rank的21个target C4层均实际走到PTO路径，未再出现共享缓存别名拒绝。

每rank每层的观察计数一致：`pto_tokens24=22`、`pto_tokens18=1`，
即S6的B4与B3调用；同时`native_tokens6=8`、`native_tokens1=1`，
保留Native的非PTO派发，不能把整条模型链表述为完全由PTO执行。
全部rank-layer累计7728次PTO CSA调用。未新增生产代码或修改依赖实现。

直接比较已有`decode_native_b4_v1`与本轮相同rank/case/request的输出token列表，
64/64请求逐token完全一致，8192个生成token无差异。该结果是本次短历史场景的
整模型生成结果对照，不代替各层张量精度、长历史、其他batch及完整P3/P4验收。

结果汇总`decode_pto_b4_updated_v4/summary.json`，明细为16份rank*.json和日志。
本轮elapsed包含首次编译、离线缓存IO及观察hook，不据此给出稳态吞吐或延迟结论。
下一步尚需有统一warmup与计时范围的Native/PTO性能对照，以及计划内其他离线场景。

## 94. 2026-09-23：提交本轮验证并整理后续session交接

按用户要求，本轮测试工具、失败定位、依赖安装记录和D16成功证据已用详细中文说明
提交为`f9bdbb5`并推送至`dsv4-flash-pto-v0.25.1rc1`。全部105个文件位于
tests/pypto_test，没有新增生产源码改动；大权重、缓存快照和二进制未提交。
根目录build_output的16份JIT产物及重复CPU编译中间文件保留本地，路径/大小已记入
LOCAL_ARTIFACTS.json，没有新增hash校验，也没有重跑测试或提交检查。

另建`DSV4_FLASH_CSA_NEXT_SESSION_HANDOFF.md`，独立整理环境与本地依赖差异、
当前可用缓存、已完成证据、稳态性能与长场景/多batch待办、原P5缺口，以及仍须
保持暂停的P3/P4、DP padding和剩余精度排查。文档提供可复用命令和旧脚本适配注意，
避免新session重新恢复已经可用的环境，或将旧基线PASS和本轮短场景结果外推为完整验收。

## 95. 2026-09-23：新基线 Native/PTO profiling 对照与 PTO 泳道图

用户要求两份可对照的 PyTorch profiling（不带 Python stack、含 device kernel）和一份 PTO 泳道图。
离线 P/D 工具新增 `profile`、`profile-export`、`profile-compare`、`swimlane`、`swimlane-export`
五个入口，全部位于 `tests/pypto_test/offline_pd/`，没有为诊断改动生产 CSA 实现。
泳道所需的 `pypto.torch.init` 诊断参数，由 worker extension 在模型加载前按环境变量补入，
只作用于被选中的那个 rank；`init` 的诊断配置只能在首次调用时绑定，故不能改为运行中开启。

三次真机运行均使用正式 ModelSlim W8A8、已有 `smoke_bank` 的 h255 输入、每 rank batch4、eager：

| 运行 | 任务 | 终态 |
| --- | --- | --- |
| Native profiling | `task_20260923_192802_7342913133` | exit0 |
| PTO profiling | `task_20260923_193343_85100714792` | exit0 |
| PTO 泳道采集 | `task_20260923_194020_99836730091` | exit0 |

两次 profiling 各先跑一轮 96 token 预热，排除首次编译与缓存冷读，再在第 8/9/10 个稳态
decode step 上开窗；两侧窗口每步都是 4 请求 24 token，构成完全一致。本轮 PTO 与 Native
的 4×128 个输出 token 逐 token 相同。采集为 CPU+NPU、Level1，record_shapes、profile_memory、
with_stack、with_modules 全部关闭。进程内解析未生成 `ASCEND_PROFILER_OUTPUT`，改由
`profile-export` 对同一批原始数据离线解析补出，未重跑真机，原始 PROF 记录未修改。

rank0 三步窗口的 device kernel 汇总（微秒；含 EP 等待、采集与同步开销，不是稳态性能结论）：

| 项 | Native | PTO |
| --- | --- | --- |
| device 记录条数 | 7647 | 5568 |
| device 合计 | 1114786.6 | 1452000.0 |
| MIX_AIV | 992477.3 | 1279666.6 |
| MIX_AIC | 61601.9 | 83811.8 |
| AI_CPU | 1270.9 | 43209.9 |
| AI_CORE | 27084.2 | 19737.8 |
| AI_VECTOR_CORE | 32352.3 | 25574.1 |

窗口内 CSA 调用为 3 step×21 层＝63 次。PTO 侧恰好出现 63 次
`aicore_kernel_mode_0_mix_aic`（41404.1）与 63 次 `simpler_aicpu_kernel_exec`（41663.8），
即每次 CSA 调用一个 AICore kernel 加一个 AICPU 任务，与"一次完整 PTO 提交"的实现一致。
被吸收的 Native 算子调用数差值均为 63 的整数倍：`VllmQuantLightningIndexer` 63→0，
`SparseAttnSharedkv` 与 `TransposeBatchMatMul` 各 −63，`Compressor` −126，
`QuantMatmulV5` 与 `DynamicQuantV2` 各 −189，`ScatterNdUpdateV2` 与
`InplacePartialRotaryMul` 各 −252，`Matmul` −315。这些减少合计约 35.9 毫秒。

即在 B4/S6/H255 这一档，每次 CSA 调用 Native 约 570 微秒的 device 算子，PTO 为约 657 微秒
AICore 加约 661 微秒 AICPU。AICore 与 AICPU 属不同执行道，两者不能相加当作延迟；
是否落在关键路径需结合泳道判断。差值最大的单项是 `MoeDistributeDispatchV2`
（892722.6→1197223.7，+304501.1），它是 EP 全互联算子、其 device 耗时包含等待对端，
在没有进一步证据前不能归因为 CSA 计算变慢，这是下一步首要待查项。

泳道采集在 rank0、`model.layers.2.self_attn.attn`、24 token 的真实调用上开了且只开了一个
DFX 窗口。一次 CSA 调用含 810 个 AICore 任务、46 个命名 callable，跨度 617.80 微秒；
每任务平均执行 18.10 微秒、平均 dispatch→finish 33.68 微秒，执行占比 53.76%。
注意力主体利用率高：`qk_pv_aiv` 48 个任务平均 70.06 微秒、执行占比 93.4%，
`qk_pv_aic` 24 个任务平均 68.51 微秒、占比 94.7%。停顿集中在小任务：
`merge_norm` 48 个任务执行占比 13.6%，头部开销 73.91 微秒中 NoC 传播占 64.55；
`weights_proj_reduce` 占比 5.4%；`qr_rms_norm_quant` 占比 9.6%，本地 dcci+ack 达 60.12 微秒；
`indexer_boundary_init` 14.3%、`indexer_head_coefficients` 16.6%、`quant` 20.3%、
`csa_cache_writeback` 22.1%、`proj_b_act` 23.4%。DFX 带边界同步与诊断开销，
只用于查看任务与依赖，不参与耗时对比。

本轮两次 profiling 的预热轮耗时为 Native 17.8 秒、PTO 60.0 秒（各含自身首次编译），
与此前 D16 整轮观察方向一致；但两者都不是稳态计时，不能据此给出加速比或回归结论。
旧单层 B40/H131073 ACL Graph 对照是另一档配置，其比值不能外推到本轮 B4/H255。
稳态性能对照（A1）仍未开展。

产物在 `results/release_csa_profile_20260923/`：`profile_comparison.json` 为对照汇总，
两份 `trace_view.json` 分别为 47844892 与 36488934 字节，连同 16 rank 的 PROF 原始数据
共约 829MB 留本地，路径与大小见该目录 `LOCAL_ARTIFACTS.json`；泳道产物
`swimlane/swimlane/merged_swimlane.json` 可直接拖入 Perfetto 打开。未做任何 hash 校验。

## 96. 2026-09-23：PTO 慢在主机侧，设备大部分时间空闲

承第95节，继续用同一批已采数据做纯 CPU 分析，未重跑真机、未占用设备队列。

`step_trace_time.csv` 显示两侧的设备占用差别极大（三个 step 窗口，微秒）：

| 项 | Native | PTO |
| --- | --- | --- |
| Computing | 1105701.9 | 1407207.5 |
| Free（设备空闲） | 92482.9 | 5345275.7 |
| Stage（总跨度） | 1198184.8 | 6752483.3 |

即每步 Native 约 399 毫秒、空闲 31 毫秒；PTO 约 2251 毫秒、空闲 1782 毫秒，空闲占 79%。
对 `Ascend Hardware` 轨道合并忙区间后统计空档：Native 最大空档 11.4 毫秒，超过 20 毫秒的
空档为 0；PTO 有 63 个超过 20 毫秒的空档、合计 5204 毫秒，单个约 90 毫秒。
63 恰为 3 step×21 层，即每次 CSA 调用对应一个设备空档。

把这 63 个空档与主机事件对齐，全部被同一条调用链覆盖：
`vllm::dsv4_csa_forward` → `dsv4_csa::attention` → `dsv4_csa::_pypto_attention_mutate`，
63 次合计 5978 毫秒，平均每次 94.9 毫秒。同一窗口内的 AscendCL API 合计仅 5 毫秒，
占 0.1%，其中最多的是 567 次 `aclrtRecordEvent`（2.5 毫秒）和 1008 次
`aclrtSetCurrentContext`（0.5 毫秒）。也就是说这 94.9 毫秒既不是 kernel 下发、
也不是等待设备，而是 PyPTO 算子内部的纯主机侧工作。

对照设备侧：同一次调用的 AICore kernel 约 657 微秒，主机与设备之比约 144 比 1。
本轮测量轮 16 个 rank 完全一致：Native 10.2 秒、PTO 61.7 秒，各 22 个稳态 step，
即约 462 与 2805 毫秒每步。每步 21 次 CSA 调用的主机开销 21×94.9＝1993 毫秒，
可解释两者每步 2343 毫秒差值的约 85%。

由此修正第95节留下的待查项：在设备空闲 79%、每个 rank 都按相同节奏停顿的前提下，
`MoeDistributeDispatchV2` 多出的约 304 毫秒与"EP 全互联算子吸收跨 rank 偏斜等待"
一致，没有证据表明通信本身变慢。这不是最终归因，但优先级应让位于主机侧开销。

需要明确的边界：本轮是每侧一轮、预热一轮后的单轮对照，不是 A1 的稳态计时协议，
没有 p50/p95 与显存数据；三步采集窗口本身为 2251 毫秒每步，低于整轮均值，
说明该结论不是 profiler 开销造成的。进一步定位需要主机侧函数级证据，
当前 trace 按用户要求关闭了 Python stack，无法在 `_pypto_attention_mutate` 内部再细分。
相关实现位于 PyPTO 启动路径，按用户约束不自行修改其运行时实现。

## 97. 2026-09-23：主机侧开销定位到 PyPTO 每次调用重复遍历 AST

承第96节。第95～96节的 trace 按用户要求未采 Python stack，无法在
`_pypto_attention_mutate` 内部细分，故新增 `hostprofile` 入口：在稳态 step 上开
cProfile，窗口构成判定与 NPU 采集一致。任务 `task_20260923_195929_134956311463`
exit0，同样使用正式 W8A8、smoke_bank 的 h255、每 rank batch4、eager，预热一轮后
对第 8、9 两个稳态 step 采样，共 42 次 CSA 调用；测量轮 62.5 秒，与第95节的 61.7 秒
同量级，说明 cProfile 只作用于 22 步中的 2 步，未显著改变整轮。

rank0 采样内的 PyPTO 调用链（cumtime 秒 / 调用次数）：

| 函数 | cumtime | 次数 |
| --- | --- | --- |
| `jit/decorator.py:3374(__call__)` | 13.35 | 42 |
| `jit/decorator.py:3139(_resolve_compiled)` | 13.33 | 42 |
| `jit/decorator.py:2707(_get_source_hash)` | 6.80 | 42 |
| `jit/decorator.py:902(_constant_dependency_names)` | 6.57 | 1764 |
| `jit/decorator.py:2735(_resolve_constexpr_bindings)` | 6.08 | 42 |
| `jit/decorator.py:1942(_expand_constexpr_variants)` | 6.08 | 42 |
| `jit/decorator.py:1740(_dep_call_nodes)` | 5.81 | 1764 |

即每次 kernel 调用都进入 `_resolve_compiled`，其占整个调用的 99.8%。它内部沿两条路径
重新遍历各子函数的 AST：`_get_source_hash` → `_constant_dependency_names`，以及
`_resolve_constexpr_bindings` → `_expand_constexpr_variants` → `_dep_call_nodes`。
1764＝42 次调用×42 个子函数，与该 CSA kernel 的 46 个命名 callable 量级一致。
自耗时最高的是 Python 标准库 `ast.py`：`iter_child_nodes` 3.65 秒／7290864 次、
`iter_fields` 2.06 秒／8668506 次、`walk` 1.95 秒／3699864 次，`ast.walk` 累计 10.58 秒、
占 kernel 调用的 79%。

两处遍历的输入都只有函数对象（以及 `_dep_call_nodes` 的依赖调用名），源码在运行期不变，
结果可按函数缓存。也就是说，这部分开销来自缓存键的重复推导，而不是编译、
描述符校验或 kernel 下发；与第96节"窗口内 AscendCL 仅占 0.1%"一致。

量级互相印证：cProfile 下每次 CSA 调用约 318 毫秒，第96节无 cProfile 的 NPU trace 为
94.9 毫秒，比值约 3.4 倍，属 cProfile 对 Python 密集路径的正常放大。

边界：cProfile 放大 Python 调用开销，上述秒数只用于相对归因，不能与设备耗时相加，
也不是稳态性能结论。相关实现位于 PyPTO 的 JIT 装饰器路径，按用户约束未自行修改，
也未验证任何修复方案的效果。证据为
`results/release_csa_profile_20260923/hostprofile_pto/host_hotspots.json`
及同目录 16 份 `rank*.prof` 与 `rank*.hostprofile.json`。

## 98. 2026-09-24：第95～97节结论的适用范围限于 eager，不代表上线路径

T2.2。第95～97节的三轮测量（`task_20260923_195243_*`、`task_20260923_195929_134956311463`
等）全部以 `--graph-mode eager` 运行——当时本机图模式根本起不来，
`aclnnAddRmsNormBias` 在基础 CANN 9.0.0 的 `libopapi.so` 和已构建的 CSA 自定义算子包
里都不存在，`norm_quant` 融合 pass 的 pattern 被 PyTorch 以 `tracing_mode="real"`
追踪时会真的执行一次，图编译在建 pattern 阶段即崩。该阻塞已于 2026-09-24 通过
配置开关 `ascend_compilation_config: {fuse_norm_quant: False}` 绕开，
`task_20260924_001111_370250932735` 是本工作区第一次跑通图模式。

因此需要明确标注：

- **上线口径是 ACL Graph `FULL_DECODE_ONLY`，不是 eager。** eager 每步都重新进入
  Python 派发路径，图模式下 decode step 捕获一次之后只做重放，
  `model_runner_v1.py` 里那句 "Python forward gates do not run during graph replay"
  就是这个意思。
- 所以第97节"每次 kernel 调用都进入 `_resolve_compiled`、占调用耗时 99.8%"
  是 **eager 特有现象**：那条路径按调用次数计费，而图模式下它只在预热与捕获时走一遍，
  不随 decode step 累积。把第95～96节"PTO 慢在主机侧、设备大部分时间空闲"的结论
  搬到生产配置上是不成立的。
- 同理，第95节的 Native/PTO 每步耗时对照也只是 eager 下的结构对照，
  不能当作上线性能差距。

**一个尚未证实的前提**：上述推理成立的条件是 PTO 的 kernel 下发本身可被图捕获。
截至本节，图模式跑通的那一轮用的是 `--backend native`；PTO 在 `FULL_DECODE_ONLY`
下的首次运行正在验证中（见 T1.4）。在拿到该结果之前，不要把"图模式下 PTO 主机开销
消失"当作已确认的结论，只能说"eager 下的归因不适用于图模式"。

**另一处偏离上线口径**：本机关闭了 `fuse_norm_quant`，而参考脚本所在环境具备该算子、
融合是开启的。因此 T2.1 之后给出的图模式性能数字同样不直接等同于线上，
必须随数字一并注明这一点。

T2.5（是否处置 PyPTO 的 `_resolve_compiled` 重复遍历 AST）不受本节影响，
仍按约束不自行修改，待用户决定走上游还是本地方案。

---

<a id="log-20260926"></a>

## 99. 2026-09-26：按新合同重构清单，修正验证入口并固定工具链

本节起记录本轮实际执行过程。工作目录为
`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`，
分支 `dsv4-flash-pto-v0.25.1rc1`；本次补记时实现提交为 `f2f54b96`。
正式模型固定为 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 的 75 分片 ModelSlim 权重，
环境为 A3、CANN 9.0.0、vLLM 0.25.1、vLLM-Ascend 0.25.1rc1，
命令先加载工作区 `env-dsv4-0251rc1.sh`，不混用其他权重或 Python 安装。

用户确认后，`bbffc888` 单独提交重构后的执行清单，`b1ed20f0` 补充所有测试先尽量单卡、
再做正式权重 16 卡的规则。精度版和性能版分别验收；允许有明确规则且误差受控的量化/Top-K 差异，
仍要求整模型逐 token 和 DSpark 统计一致。两侧使用相同 mode，mode=1/2 均测后选定主口径。
最终性能目标为整模型 PTO 明确快于 Native，以及指定 B16/S6/H8192 图模式下
HC_pre→norm→CSA→HC_post 完整设备区间 <750 μs；当前没有宣布达到这些目标。
完整合同和暂停项只维护在清单中，本日志不另设验收标准。

先修正会使验证结论失真的基础问题：

| 提交 / 清单项 | 实际修改 | 已执行的验证与范围 |
| --- | --- | --- |
| `135a84f7` / A1 | NZ mode 从 CLI 传至 rank 和模型配置；拒绝环境/导入值冲突；按根签名实际布局打包并记录四张权重 | 8 项 CPU 配置回归通过；整模型配置仍待 A5 |
| `6ee2f62d` / A2 | 逐元素比较器检查 shape/dtype、有限值、必需输出和逐项容差；无参考只能 MEASURED | 11 项 CPU 回归通过；均值相同不再等于数值通过 |
| `b12d7b42` / A3 | schema=2 保存调用前初态、布局、stride/offset、共享 storage 与读写角色；回放恢复可变状态，拒绝未知旧快照 | 6 项回放回归与受影响的 11 项门禁回归通过；在线只备份已声明写入页，不据此宣称完整保护区通过 |
| `03d9da98` / A4 | 固定兼容的新版官方 PTOAS/ISA，移植 PyPTO 主线已有适配并同步运行时 | 11 项定向标量 API 回归、性能版整层 PTOAS/CCE 编译及单卡兼容检查通过 |

工具链固定为 PTOAS 0.66、官方 PTO-ISA `327cd586`、PyPTO `f9b24ceb`、
Simpler `a54c05095`；后两者仍在 `feat/kernel-mode-integration-test`。
PyPTO `63cdd96d` 移植官方主线 `d626aea1` 的标量读写适配，`f9b24ceb`
对齐 descriptor/SDK、runtime 子模块和 NPU 适配扩展；Simpler 更新 ISA pin 并重新构建运行时。
PTOAS/ISA 没有额外实现补丁。当前允许在指定 PyPTO/Simpler 调试分支做必要修改，
第 97～98 节“不自行修改”的旧约束不再代表本轮授权。
版本和验证记录见 [toolchain/validation.json](results/csa_baseline_20260926/toolchain/validation.json)。
A4 仅确认兼容性与代表路径，不等于全部 NZ、graph 或整模型验收。

## 100. 2026-09-26：建立当前 release 的单卡对照，补齐同初态和保护区检查

旧单层脚本依赖已删除的 metadata executor 和旧模块路径，因此 `45442f41` 建立
`dsv4_csa_single_layer.py` 与 `run_csa_single_layer.sh`，使用当前 Native builder、
实际物理页布局和正式模型第 2 层权重。测试构造输入与历史 cache，覆盖注意力半层及外层
HC/norm，不加载 MoE；它是定位和筛选用例，不能替代真实输入的整模型结果。
3 项存储保护 CPU 回归通过。

每个实现都从相同初态执行两次，比较 8 类输出/状态：
`x_out`、`idx_topk`、`swa.0`、`compressed.0`、`state.0`、`indexer.0`、
`indexer.1`、`indexer_state.0`。同时检查 Native metadata 未改写、未声明 slot、
页 padding、分配前后保护区和非有限值。`--save-case` 保存调用前 schema=2 快照，
Native 参考另存。随机种子为 1024；以下均为 S6/H8192，Native 开启
`set_deterministic_level(1)` 与 `HCCL_DETERMINISTIC=true`。

以下任务均通过 `task-submit` 完成；跨实现误差仍在诊断，因此报告状态都是 **MEASURED**。
保护区/metadata 检查通过，不代表已经确定浮点状态容差或通过 token/DSpark 验收。

| 单卡配置 | 任务 ID | 已保存报告 |
| --- | --- | --- |
| B4，精度版，mode=0，原默认规约 | `task_20260926_111536_305102231717` | [precision_nd_v2](results/csa_baseline_20260926/native_b4h8192_precision_nd_v2/report.json) |
| B4，性能版，mode=0，默认 atomic=1 | `task_20260926_115108_32965273337` | [performance_nd](results/csa_baseline_20260926/native_b4h8192_performance_nd/report.json) |
| B4，性能版，mode=0，固定 atomic=0 | `task_20260926_120105_333300523870` | [performance_fixed](results/csa_baseline_20260926/native_b4h8192_performance_fixed/report.json) |
| B4，精度版，mode=2，固定 atomic=0，图 A→B→A | `task_20260926_122302_351515921990` | [precision_nz2](results/csa_baseline_20260926/native_b4h8192_precision_nz2/report.json) |
| B4，性能版，mode=1，固定 atomic=0，图 A→B→A | `task_20260926_122509_35294426490` | [performance_nz1](results/csa_baseline_20260926/native_b4h8192_performance_nz1/report.json) |
| B16，性能版，mode=1，固定 atomic=0，图 A→B→A | `task_20260926_123008_355121121165` | [b16_performance_nz1](results/csa_baseline_20260926/native_b16h8192_performance_nz1/report.json) |

首轮精度版 mode=0 的 Native/PTO 各自重复均精确一致。跨实现 `x_out` 的
max_abs=0.015625，Indexer INT8 cache 有 2 个元素不同，scale 精确一致；
Top-K 有 10292 个位置不同。该轮尚未保存集合诊断，不能由位置差异直接判断是排序还是选择差异。
后续 `5cc3ff81` 增加集合/顺序区分，并保存 Native QLI 的 query、qscale、head 权重和页表；
2 项相关 CPU 回归通过。进一步结果见下一节。

最新 B16 单卡用例已完成：两侧各自重复的 8 类输出/状态全部精确一致，图 A→B→A 通过；
跨实现 `x_out` 有 379950 个元素不同，max_abs=0.03125、RMSE≈0.00186714，
Indexer INT8 cache 有 125 个元素不同，scale 有 12 个不同、max_abs=0.00006103515625。
Top-K 96 行集合不同，共替换 470 个候选，无越界、重复、缺项或错误 padding。
这些是误差测量，不是允许阈值；尚未证明这些差异不影响模型生成。

## 101. 2026-09-26：增加固定规约诊断，区分运行间波动与跨实现差异

性能版默认路径在同输入、同初态下仍有运行间波动。因此 `ecccc02ba` 增加共用
`reduction.py` 和集中定义的 `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0/1`，默认 1。
配置必须在导入/编译前固定，初始化拒绝导入后切换；不改变图重放期间的根 ABI。
0 将 QR/KV 改为单 K 分片、单写入者，按固定 K 块顺序累加并使用非 atomic store。
这也改变累加分组，不能把它当作默认路径的逐 bit 参考或默认部署性能结果。

覆盖的 store 点为精度版 QR 2 处/KV 2 处，以及性能版 QR ND 2 处/QR NZ 2 处/KV 2 处。
原默认 split-K 分别为精度版 QR=1/KV=2、性能版 QR=8/KV=8，关闭时均为 1。
3 项 CPU 配置回归和性能版整层 CPU lowering 通过，关闭时生成 IR 无 atomic store。
B4、mode=0 的同初态比较如下：

| 性能版配置 | 自身重复的 `x_out` | 自身重复的其他状态 | 跨 Native/PTO Top-K 集合 |
| --- | --- | --- | --- |
| atomic=1 | 283 个元素不同，max_abs=0.0078125 | SWA 1 个元素不同，max_abs≈2.98e-8；其余一致 | 24 行共替换 115 个候选 |
| atomic=0 | 精确一致 | 全部精确一致 | 24 行共替换 116 个候选 |

Native 两组均自身精确一致；关闭 PTO atomic 消除了本例观察到的运行间波动，
但没有消除跨实现差异。不能据少量重复推断所有形状的全局确定性。
两组 Native 参考和初始快照已确认相同，固定规约报告复用默认组的 `case/` 和
`native_reference.pt`，删除重复副本，保留不同的 PTO Top-K 结果。

为判断候选替换来源，复用已保存 Native QLI 入参做 CPU 算术分析：
按 Native 的 QK/1024→FP16、FP16 head 系数、head 归约和 key scale 公式计算，
24 行选中集合全部与 Native 相同；保持 Native 输入、仅换成性能版 FP32 评分公式时，
总共替换 9 个候选。整条性能版链替换 115 个，说明不能把剩余差异全部归于评分精度策略，
仍需定位上游查询投影/量化与 weights 投影。
分数比较将 Native 分数乘 1024 统一量纲；CPU sum 不宣称复刻 Cube 的逐 bit 累加序。
记录见 [topk_boundary.json](results/csa_baseline_20260926/native_b4h8192_performance_nd/topk_boundary.json)。
量化/Top-K 是否可接受仍须满足预先明确的误差规则以及整模型 token、DSpark 统计一致。

## 102. 2026-09-26：修复实际 NZ 配置验证与 Native 能力限制，完成代表形状的图内容更新

单卡用例最初未在权重加载前设置 `torch.npu.config.allow_internal_format=True`，
会出现名义 mode=2 而实际仍为 ND 的无效 NZ 记录。`f2f54b96` 将此设置与 Native runner 对齐，
并强制检查四张目标权重和 Compressor 权重的实际格式；该名义 NZ 记录已撤回并清理。

真正启用 NZ 后，单卡先后暴露两个 Native 功能问题，`8ab5e92c` 分别处理：

1. 当前 CANN 融合 Compressor 要求 `wkv/wgate` 为 ND，加载期显式保留 ND，避免全局 mode=2
   将不支持的权重转换为 NZ。
2. 当前 CANN 9.0 的 `libopapi.so` 有 `TransposeBatchMatMul`，没有
   `TransposeBatchMatMulWeightNz` 及其 workspace 查询符号。非 A5 的 Native `wo_a`
   在缺少该能力时显式保留 ND 并记录原因；有官方能力时继续走 NZ，避免 decode 热路径转换。

4 项 Linear CPU 测试、3 项算子符号能力 CPU 测试及改动文件 Ruff 检查通过。
这些 Native 回归不依赖 PyPTO。两个失败重试的冗余产物已清理，保留修复后有效报告。
上述 ND 例外不撤销 PTO 四张权重 NZ 的目标；mode 相同也不意味着两侧每张权重布局相同。

当前真实布局如下，顺序固定为 `wq_a / wq_b / wo_a / wo_b`：

| 用例 | Native 实际布局 | PTO 根签名布局 |
| --- | --- | --- |
| 精度版 mode=2，B4 | NZ / NZ / ND / NZ | ND / ND / NZ / ND |
| 性能版 mode=1，B4 与 B16 | ND / NZ / ND / NZ | ND / NZ / ND / ND |

上述三个用例均在 atomic=0 下完成 PTO 同地址 A→B→A 图重放：B 使用与 A 不同的输入内容，
先取得相应 eager 参考，每次重放前恢复初始 cache/state，8 类输出/状态与对应参考精确一致，
metadata 与保护区检查通过。这只验证固定形状、固定 metadata 的输入内容更新；
padding、档位/metadata 变化、多 leaf 及整模型图行为仍待验证。

精度版 mode=2 的跨实现输出 max_abs=0.015625；Top-K 有 23 行集合不同、82 个候选替换，
另 1 行仅顺序不同。性能版 mode=1 的 B4 有 24 行集合不同、116 个候选替换；
B16 结果见第 100 节。各报告仍为 MEASURED，不能标注数值验收通过。

## 103. 2026-09-26：清理记录的保留例外与当前未完成事项

`30795c69` 删除了依赖旧 executor/模块路径的用例、失效计划、重复快照、旧 profile 和失败重试产物，
其中错误地一并删除了本日志。用户明确要求长期保留本文件并继续记录当前过程，
本次从 `30795c69` 的父提交恢复删除前完整原文：3277 行、第 1～98 节均保留，
再追加第 99～103 节；清单、README 和根目录接续入口同步注明这个保留例外。
历史结论限定在对应版本与场景，原始历史链接允许指向已清理产物的 Git 版本；
不因恢复日志而恢复过时测试入口或已撤回的验收结论。

继续保留当前单卡报告和可复用快照、7 组正式权重 bank、工具链验证、尚未关闭的
A3 TDIV 原始复现证据。TDIV 旧证据对应当时工具链，不能说已在 PTOAS 0.66 上复验。
本次恢复与补记只读取既有报告、核对文档和 Git 内容，不做 hash 扫描或新增设备测试。

截至本次补记：

- A4 的兼容性验证已完成；A1～A3 有 CPU 和代表单卡证据，A5 仍在进行中，整模型出口未完成。
- B1 已有可关闭 atomic 的诊断路径及代表单卡重复/图检查；B2/B3 的跨实现误差仍须定位，
  各浮点状态和量化/Top-K 规则尚未全部确定。
- 四张 PTO 权重 NZ 尚未全部接通；不能把当前部分 NZ 布局写成“四张完成”，ND 路径继续保留。
- 已准备正式 H8192 bank、B16/TP1/DP-EP16、同 mode=1、FULL_DECODE_ONLY、
  96 个 decode token 的 Native/PTO 基线命令；此轮 **尚未提交或运行 16 卡任务**。
  计划先用 Native 确定性与 PTO atomic=0 检查逐 token/DSpark，再按差异补最小诊断。
- 部署性能配置、mode=1/2 的主口径选择、完整设备区间和整模型加速均未验收；
  当前没有本轮 <750 μs 的证据。后续继续按清单依赖推进，每个阶段在本日志追加结果与限制。

## 104. 2026-09-26：完成当前 mode=1 固定规约的正式权重 16 卡 token/DSpark 基线

第 100 节 B16 单卡先完成后，提交任务 `task_20260926_124315_37151566340`；
该任务顺序运行 Native、PTO，最终 exit=0。实现源码固定为 `c7d08cf0`，
PyPTO/Simpler/PTOAS/ISA 与第 99 节一致。任务期间未修改其 JIT 读取的源码。

两侧使用同一正式 H8192 bank，B16/TP1/DP-EP16、FULL_DECODE_ONLY、capture size=96、
每请求生成 96 token，关闭 KV NZ；PTO 为性能版、atomic=0、QR/KV split-K=1。
两侧开启 Native level=1 确定性与 `HCCL_DETERMINISTIC=true`，保留 HCCL AIV；
受当前 CANN 算子能力限制，`fuse_norm_quant=False`，draft 保持 eager。
这是正确性诊断配置，不作为部署性能数字。

新增纯 CPU 入口 `offline_pd/compare.py`，按 bank 预期 case 和显式 rank/batch/token 数检查完整性，
逐 token 比较，并检查 DSpark 草稿数、草稿 token 数、接受总数和逐位置接受计数。
缺文件、缺统计、长度不符或任何差异都失败；PASS 的范围明确不含层级误差、保护区和性能。
4 项针对性 CPU 回归通过（0.03 秒）：token 换位、接受总数相同但逐位置分布不同、
缺 rank、缺 DSpark 计数；新文件 Ruff 检查通过。

| 项目 | 本轮实际结果 |
| --- | --- |
| 覆盖 | 16 个 rank × 16 个请求 × 96 token，共 24576 token；bank 的 4 个输入变体均覆盖 |
| 逐 token 对照 | 0 个差异 |
| DSpark 对照 | 16/16 rank 完整且一致；每 rank 为 drafts=271、draft tokens=1355、accepted=1280，逐位置均为 `[256,256,256,256,256]` |
| 配置贯通 | 全部 rank 的 mode 请求值/环境值为 1，记录 level=1 与 FULL_DECODE_ONLY；PTO 固定规约日志完整 |
| PTO 选择 | 每 rank 的 21 个目标 CSA 层均有捕获期 `pto_tokens96` 命中；PTO 根布局为 ND/NZ/ND/ND |
| 验收边界 | 本轮 token/DSpark PASS；未采逐层误差或稳态设备区间，不能外推 mode=2 或默认 atomic 路径 |

证据目录为 `results/csa_baseline_20260926/model_b16h8192_nz1_fixed/`：
[manifest](results/csa_baseline_20260926/model_b16h8192_nz1_fixed/manifest.json)、
[逐 token/DSpark 比较](results/csa_baseline_20260926/model_b16h8192_nz1_fixed/comparison.json)、
[配置与捕获期层选择](results/csa_baseline_20260926/model_b16h8192_nz1_fixed/execution_checks.json)，
同目录保留两侧 `rank*.json`、运行命令和本地原始日志，不提交重复编译目录与二进制。

下一步继续 C 的四张 NZ：上游 PyPTO `8a944cf2` 已补充 NZ 偏移的除法/取模非负证明、
leading-axis slice 和 dispatch 支持，`1d7890e9` 修正 NZ 参数的逻辑 stride 标注。
只读检查发现整提交回移有上下文冲突，尚未将其应用到当前调试分支；后续按需要移植、
先 CPU 编译与针对性回归，再单卡验证。A5 的层级误差和部署性能仍未完成。

## 105. 2026-09-26：移植主线 NZ 能力，接通四张根权重并共用适配

本节继续第 104 节的 NZ 工作，不修改其旧版本 16 卡结论。PyPTO 调试分支提交
`712adef8` 移植 `8a944cf2`、`1d7890e9`、`0c8a2753`，补齐其依赖的 slice 布局传播、
错误类型及绑定；保留本分支的 schema=3 kernel ABI 和 formal Out 约束。
Simpler 仍为 `a54c05095`，官方 PTOAS 0.66、PTO-ISA `327cd586` 均未加实现补丁。

移植验证中先发现 CMake build 不会更新环境里已安装的 extension，改为当前工作树 editable
重装；随后发现少了一项上游 slice 布局传播前置改动并补齐。最终 114 项 NZ/layout/dispatch
及 formal Out CPU 回归通过（6.18 秒），仅运行受影响筛选项。
命令与最新输出记录在 `results/csa_baseline_20260926/nz_upstream_port/`，不保留重复失败副本。

算子侧改动：

1. 精度版补全 `wq_a/wq_b` NZ 注解，性能版启用已有的 QR NZ 子函数。
   精度版原先的反向 K 索引在条件分支合流后无法证明非负；单纯等价的正数取模表达仍无法
   消除该合流限制。最终保留原 K 顺序，只在已知属于 `[0,N)` 的块索引上写 `max(index,0)`，
   这是有效输入域内的恒等式，不调整累加树。
2. `wo_b` 从 `[D,G*K]` 改为 PTO 私有 `[G,D,K]`，主机只在加载/回放准备期重排；
   编排侧取 `wo_b[g]` 完整平面，核内只处理矩阵块索引。ND/NZ 子函数均保留，
   INT32 group partials 及两版各自量化/反量化策略不变。
3. 性能版适配改为精度版共用实现的薄入口，显式传入性能版根函数和 kernel。
   权重准备、metadata/cache 绑定、参数表生成不再维护两份；算术差异继续留在各自算子中。
4. schema=2 回放读取目标根形状：旧二维权重先按来源 ND/NZ 解包，再转分组视图，最后按目标
   根布局打包；相同布局且相同形状保留原存储，共享存储或非只读转换仍拒绝。
   新增 3 个分组回放回归，以原二维 INT32 matmul 对照转换后分组 matmul；配置/回放共 17 项通过。

当前两版布局一致：mode=0 全 ND，mode=1 仅 `wq_b/wo_b` NZ，mode=2 四张全 NZ。
两版 mode=2、atomic=1 的完整层均完成 CPU lowering、PTOAS 0.66 与 CCE 编译，
生成设备二进制，未初始化 NPU：
当时两版的临时编译报告已由第 107 节 Native 存储方案替代并清理。
改动文件 Ruff 与 Git 空白检查通过。单卡增加可选 `--save-state`，便于逐元素比较布局变化。

本节提交时设备验证尚未运行。下一步先 B16/S6/H8192、atomic=0 的 mode=0/2 单卡对照，
检查两版各自全部 8 类逻辑状态、保护区与图内容更新；随后量部署路径和 mode=1/2 收益。
不能用编译 PASS 宣称数值通过、四张 NZ 已验收或已达到 <750 μs。

## 106. 2026-09-26：按用户反馈核对 Native / pypto-lib NZ，撤回“必须 CPU 重排”

用户指出第 105 节实现沿用 CPU 重排不符合 Native 接入预期。本轮停止该方案的提交与整层上卡，
先读当前 Native、pypto-lib `2164563`、PyPTO torch 桥接及 CANN 头文件，再用最小单卡核对。
这是一项实现判断的更正，不把原错误说法继续保留为当前指导。

源码依据：Native `utils.py:maybe_trans_nz` 调用 `torch_npu.npu_format_cast(weight,29)`；
`w8a8_dynamic.py:process_weights_after_loading` 先转为量化 matmul 需要的 K×N 逻辑矩阵。
pypto-lib `models/deepseek_v4_flash_dspark/utils.py:pack_nz` 将最后两维按
`[C/c0,R/16,16,c0]` 排列，c0=32/element_size；其 `TensorSpec` 是 host 初始数据用法，
不能推导出已在 NPU 的生产权重必须搬回 CPU。
PyPTO `torch/interop.py:_describe_tensor` 当前只接受格式 0/2，不支持直接传格式 29；
而 `pl.NZ` 是现有 GM 字节布局声明，不负责格式转换。

先前注释把格式 30 错写成 FRACTAL_NZ，现已删除：当前 CANN/torch_npu 的
FRACTAL_NZ=29、NCDHW=30。格式 30 被桥接拒绝不能证明 NPU 上无法完成打包。
不能笼统说 Native NZ 和 PTO NZ 的物理规则不同：对于本 A3 对齐 BF16/INT8 小用例，
两者物理字节已实测相同；必须分清张量描述符、矩阵方向/分组和物理排列。

四张权重在当前 TP1 接入中的逻辑合同：

| 权重 | Native 加载后逻辑形状 | PTO / pypto-lib 根逻辑形状 | 加载期处理 |
| --- | --- | --- | --- |
| wq_a | `[1024,4096]` | `[4096,1024]` | 转置后按根布局准备 |
| wq_b | `[1024,32768]` | `[1024,32768]` | 方向相同；当前桥接不直接接收格式 29 |
| wo_a | `[8,4096,1024]` | `[8,1024,4096]` | 转置最后两维后准备 |
| wo_b | `[8192,4096]` | `[8,4096,1024]` | 转置、按 8 组重排后准备 |

最小任务 `task_20260926_133140_401449515104` completed/exit=0：BF16/INT8 ×
`[32,64]` / `[2,32,64]`，共 4 档。在同一 NPU 上执行 Native 格式 29 转换及
原版 pypto-lib pack_nz；使用 ACL D2H 只读原始物理字节（不用会自动解码格式的 Tensor.cpu()
代替物理比较）。Native 原始 NZ 字节与 CPU golden 精确一致，NPU 打包也精确一致。

随后修正共同适配：pack/unpack/group 在输入所在设备完成，按 pypto-lib 的公式重排；
使用 Native 同一 `npu_format_cast` API 在 NPU 上明确转换基础格式，不再拷贝权重到 CPU。
仍仅在加载/回放准备期执行，不进 decode/graph replay 热路径。
任务 `task_20260926_133444_402672016789` completed/exit=0，4 档实际调用修正后的适配，
Native→PTO 打包、解包往返、转置、wo_b 分组均与 pypto-lib 精确一致，最终格式均为 2。
修正后只重跑受影响的打包/分组回放 CPU 5 项（10.27 秒），通过；相关 Ruff 通过。

证据：[基础合同](results/csa_baseline_20260926/nz_layout_contract/report.json)、
第 107 节已替代并清理该 device 重打包实验记录；当前只保留最终方案的命令。验证范围是本 A3 的对齐 BF16/INT8 小张量；整层输出、收益仍待下一步。
可复用 Native NZ 存储是后续优化候选，须同时满足根矩阵方向及 PyPTO 桥接合同，
不能直接用相同的“NZ”名称推导任意权重可零拷贝。


## 107. 2026-09-26：复用 Native NZ 原始存储，核对两侧消费语义

用户进一步指出：相同逻辑矩阵的 Native 格式 29 与 pypto-lib pack_nz 字节相同，
因此接入不应为已有 NZ 权重额外重排。第 105/106 节的私有转置/分组与 device 重打包方案
被本节替代；其临时编译目录和适配实验产物清理，不作为当前实现指导。

### 107.1 消费端核对与实现选择

Native `w8a8_dynamic.py:process_weights_after_loading` 先把量化权重变成连续 `[K,N]`，
再调用格式 29；`apply` 直接把它交给 npu_quant_matmul。当前 CANN 9.0 的
`quant_batch_matmul_v3_bf16.h` 按 `bTrans` 区分 NZ 的 `[k1,n1,n0,k0]` 与
`[n1,k1,k0,n0]` 读取，调用 `SetTensorB(..., bTrans)`；并不是另一种 NZ 打包规则。
Native `wo_a` 加载为 `[G,K,N]`，消费端 transpose_batchmatmul 使用 `perm_x2=(0,1,2)`。

pypto-lib `2164563` 的 `wo_b` 准备则先把 `[N,G*K]` 权重 reshape/permute/contiguous
成 `[G,N,K]`，再逐组 pack_nz；算子切片为 `wo_b[g:g+1,n0:n0+NT,kb*KT:(kb+1)*KT]`，
配合 `b_trans=True`。NZ 的物理规则相同，送去打包的矩阵方向和分组不同。
不能把 Native `[8192,4096]` 的 NZ 直接改标签为 `[8,4096,1024]` 而复用原切片。
当前 Native wo_b 实测 storage shape 为 `[128,512,16,32]`；生成 PTO GlobalTensor
使用等价 `[1,128,512,16,32]`，在原矩阵 K 轴选择组，权重地址不变。

当前四张 PTO 根几何全部匹配 Native：

| 权重 | Native 与 PTO 的逻辑形状 | PTO matmul 读取方式 |
| --- | --- | --- |
| wq_a | `[1024,4096]` | N×K，b_trans=True |
| wq_b | `[1024,32768]` | K×N，b_trans=False |
| wo_a | `[8,4096,1024]` | 取 group 后 K×N，b_trans=False |
| wo_b | `[8192,4096]` | K×N，按 g*1024 选择输入通道组，b_trans=False |

wo_b 证明失败的确切原因：Simplify 在 host 外层循环内知道 g∈[0,7]，先删掉 max(g,0)；
随后 outlining 把 g 变成核函数普通标量参数，范围信息没有随参数传入，后续 NZ 检查失败。
算子侧把 NZ matmul 定义为独立 incore 函数，在其参数边界保留非负表达式，
没有放宽 NZ 校验、修改 PTOAS/ISA 或增加权重重排。两版 mode=2、atomic=1 全链编译通过。

生产适配共用精度版 Native 绑定逻辑，性能版仅传入自己的算术 kernel。
已匹配格式的四张权重直接 detach 借用存储；当前 Native wo_a 因 CANN 能力限制仍为 ND，
PTO mode=2 对这一张仅调用一次 npu_format_cast(...,29)。ND 分支保留原 Native 地址。
离线 schema=2 旧快照仅对已知的三个转置矩阵做显式迁移，不再维护私有三维 wo_b。
Native NZ 快照改为 ACL D2H 读取原始物理字节；Tensor.cpu()/storage.cpu() 会解码，不能替代。

### 107.2 PyPTO 桥接与单卡证据

kernel ABI 保留参数布局并纳入编译产物身份；ND 默认描述不变。
torch 桥接仅为显式 pl.NZ、A2/A3 只读 FP16/BF16/INT8 接收格式 29，
校验完整物理形状、零 offset、无 padding 和精确容量，直接借用原 Tensor/storage/data_ptr。
实际物理形状从 torch_npu C++ get_npu_storage_sizes 查询；初版误把 Python
get_storage_size（返回元素数）当成形状，首次小任务立即失败，已修正。
203 项 ABI/桥接 CPU 回归通过；修正真实 descriptor 查询后，仅重跑受影响的 10 项并通过。

任务 `task_20260926_140857_308912571` completed/exit=0，27.35 秒：
BF16 N×K Native F.linear、INT8 K×N Native npu_quant_matmul 与 PTO 消费同一格式 29 张量，
各自与独立 CPU 数学参考精确一致；PTO 直接 JIT、注册算子、A/B/A 图重放均通过，
权重地址保持原值，测试调用阶段禁止 Python format_cast。
用例在 PyPTO `tests/st/runtime/kernel/test_native_nz.py`，本仓保存运行命令和 JUnit 结果。

两版配置/回放 CPU 回归 17 项通过（33.72 秒）。
任务 `task_20260926_141441_5833126495` 先检查 BF16/INT8、2D/3D 的 Native 原始 NZ 字节、
pypto-lib pack_nz、设备打包与新快照还原，4 档全部精确一致；随后使用正式第 2 层权重，
B4/S6/H8192、atomic=0、Native 确定性开启，执行两版各自 mode=0/2 整层与图对照。

该任务 completed/exit=0。两版各自 mode=0/2 的同初态稳定性、A/B/A 图重放、metadata
及全部保护区通过；两版分别进行 ND/NZ 对照，Native 和 PTO 的 8 类逻辑输出/状态均逐 bit 相同。
mode=0 四张原地址全部复用；mode=2 的 wq_a/wq_b/wo_b 仍是 Native 格式 29 且 data_ptr 不变，
wo_a 从 Native ND 一次性转为 29。各单层跨实现报告仍标 MEASURED，不将 Native/PTO 原有
量化/Top-K 差异算作本次 NZ 对照 PASS，更不外推整模型 token/DSpark 或 <750 μs。

PyPTO 已独立提交 `88297437`（中文、Signed-off-by）。设备任务运行的是同一份桥接及算子
工作区代码，提交仅固化已验证内容；本轮之前的 `57aa9430` 16 卡基线仍只适用于当时配置。
本轮未启动新的 16 卡任务。下一步按 C 清单补 B16/尾块/padding/长短上下文受影响项，
量默认 atomic 与 mode=1/2 的完整区间，再进入正式整模型验收。

证据：
- [物理布局与快照](results/csa_baseline_20260926/nz_layout_contract/report.json)
- [性能版 ND/NZ 逐元素对照](results/csa_baseline_20260926/nz_native_single_card/performance_nd_nz_comparison.json)
- [精度版 ND/NZ 逐元素对照](results/csa_baseline_20260926/nz_native_single_card/precision_nd_nz_comparison.json)
- [性能版实际 Native 存储绑定](results/csa_baseline_20260926/nz_native_single_card/performance_mode2/report.json)
- [精度版实际 Native 存储绑定](results/csa_baseline_20260926/nz_native_single_card/precision_mode2/report.json)

只保留上述最新报告、复现命令和必要输入。早先错误私有形状的编译目录、失败重试 dump、
被替代的 device 重打包探针记录，以及本次完成对照后的重复 states.pt 不再保留。

## 108. 2026-09-26：第二层 metadata 复用口径的单卡设备计时与热点基线

本轮设备算术基于主仓 `2228939a`，PyPTO `88297437`、Simpler `a54c05095`、
官方 PTOAS 0.66 / PTO-ISA `327cd586`。增加计时工具，没有改变算子的量化、规约或调度。
本轮没有启动新的 16 卡任务。

### 108.1 口径与测量修正

用户确认“单卡计时按照第二层的信息来”。主入口默认 `--timing-metadata reuse`：
模拟同一步第二个 CSA 层，PTO 使用前层已经生成的 compact metadata，
Native 仍按当前生产代码逐层生成。继续固定正式 `model.layers.2` 权重与合成输入/历史；
这表达 metadata 的复用状态，不宣称使用了整模型第二层的真实激活快照。
`--timing-metadata produce` 单独记录每步首个 CSA 层的生成成本，不混入主结果。

compact metadata 是新压缩 KV 行的 RoPE cos/sin 和 cache 页/行 slot 信息；
主 Compressor 与 Indexer Compressor 各有一组。这些信息必需，但独立的生产 kernel
不是不可替代的算法要求，后续可在不改变边界、padding 和索引规则的前提下评估融合。
现有 production service 已在同一步跨 CSA 层共享它们，本轮计时按这个实际复用行为执行。

完整区间仍是 HC_pre→norm→CSA→HC_post。每次重放前在区间外恢复 cache/state 初态并
毒化输出；Native Top-K hook 只保留返回引用，不把诊断 clone 或 CPU 拷贝放入区间。
每侧 5 次预热、20 次采样；默认 atomic 部署路径、Native deterministic level=0、
HCCL_DETERMINISTIC=false。精度诊断默认 level=1 的入口保留。

最初尝试在图内捕获计时事件，小探针未揭示问题；真实整层出现 Native 2.82 μs、
PTO 29.90 μs 的固定旧值，确认当前 torch_npu 图内事件时间戳不随重放更新。
这些数值全部作废，没有作为基线保留。改用图外 Event 包住 replay，检查开始时间戳
逐次前进；使用 elapsed_time 的毫秒结果转 μs，recorded_time 仅存原始计数，不冒称纳秒。
图外事件可能包含派发留下的设备间隙，因此另采 profiler 的首末设备任务窗口核对。
profiler 自身的事件包络不作为无 profiler 性能样本。

一次重试因重复 `--save-case` 触发已存在目录保护而退出，未进入后续 PTO 测量；
复用有效输入快照后正常完成。首层测量发现 metadata 生产须明确区分，随后按用户要求
另跑第二层主口径。失效计时、失败重试与被替代报告均删除，只在本日志记录原因。

### 108.2 主结果：第二层复用 metadata

任务 `task_20260926_145850_27076428841` completed/exit=0。
单卡 B16/S6/H8192、seed=1024；两侧相同 mode，性能版 atomic=1。

| mode | Native p50 / p95（μs） | PTO p50 / p95（μs） | Native/PTO p50 |
| --- | --- | --- | --- |
| 1 | 938.68 / 943.60 | 856.82 / 874.18 | 1.096 |
| 2 | 913.02 / 922.46 | 843.53 / 869.22 | 1.082 |

独立 profiler 中，Native 各有 43 个设备任务，包含两项 CompressorMetadata；
PTO 各只有 runtime/worker 两项，未重复生产 metadata，二者有重叠不能相加。
mode=1 首末设备窗口为 Native 946.72 μs、PTO 859.68 μs；
mode=2 为 Native 919.22 μs、PTO 843.52 μs，与主采样量级一致。

两档均通过 metadata、slot 外逻辑区、物理页 padding、首尾保护区及有限值检查。
Native 此用例两轮同初态输出/状态精确一致；PTO atomic 同初态 x_out 的最大差为 0.015625，
mode=1/2 分别有 3979/1924 个元素不相同。PTO 对 Native 的 x_out 最大差均为 0.03125，
RMSE 约 0.001867。mode=1 计时图对 eager 的 Top-K 有 2 行集合不同、合计 3 个替换；
mode=2 的 3 行差异仅为集合内顺序。结构检查无越界、重复或缺失。
这些均是差异诊断，不是已声明规则/容差后的数值验收，报告保持 MEASURED。

mode=2 暂作下一轮优化候选，mode=1 保留。两档之间的差距仍受运行波动影响，
不能据此选定整模型最终主口径；当前也没有达到 750 μs。

### 108.3 首层成本和独立泳道

首层任务 `task_20260926_144823_21728414911` completed/exit=0，
PTO 图内每次生成两组 metadata，设备 profiler 确认存在两项生产算子。

| mode | Native p50 / p95（μs） | PTO p50 / p95（μs） |
| --- | --- | --- |
| 1 | 912.02 / 917.48 | 910.20 / 929.58 |
| 2 | 925.78 / 931.20 | 868.92 / 903.76 |

此表来自独立轮次，只保留首层诊断信息；不能与第二层表直接相减作为 metadata 的净成本。

DFX 任务 `task_20260926_145058_2389942171` completed/exit=0，复用相同 mode=2 输入快照，
metadata 已作为入参准备。一个完整窗口、1131 条 worker 记录，无丢失窗口；
调度到完成 837.90 μs，是独立 eager 诊断，不能替代图或整模型性能。
名称映射来自该完整层唯一 kernel_config.py，并保留源表；删除猜测最新构建和旧根入口的分支。

| 设备阶段 | 最早开始到最晚结束（μs） | 解释边界 |
| --- | --- | --- |
| HC_pre 至混合 norm | 29.42～123.40 | 多个 Vector 子步骤 |
| Indexer score | 394.90～467.26 | AIC/AIV 重叠；此前还有 QR、量化与 key 重排 |
| 稀疏 QK/PV | 502.02～652.20 | AIC/AIV 重叠，最大单 worker kernel 约 141.32 μs |
| O projection 至激活 | 691.14～835.70 | A/B 投影与分组量化流水重叠 |
| HC_post | 825.42～863.58 | 与输出阶段尾部有重叠 |

这些首末窗口不能相加；跨度与单 worker 时间的差也不能全部归为计算或某一种调度开销。
稀疏注意力与输出投影是下一轮待评估重点，先补尾块、padding 和长短上下文受影响验证，
再逐项改动、比较数值和完整第二层区间，不用局部核变快替代整模型验收。

证据与复现：
- [配置、任务及版本](results/csa_baseline_20260926/nz_native_b16_timing/manifest.json)
- [主对照、首层分项与泳道汇总](results/csa_baseline_20260926/nz_native_b16_timing/comparison.json)
- [第二层 mode=1 报告](results/csa_baseline_20260926/nz_native_b16_timing/following_mode1/report.json)
- [第二层 mode=2 报告](results/csa_baseline_20260926/nz_native_b16_timing/following_mode2/report.json)
- [单窗口完整泳道](results/csa_baseline_20260926/nz_native_b16_timing/pto_mode2_swimlane/dfx/merged_swimlane.json)

CPU 汇总脚本从完整采样、CSV 及 DFX 原始记录重建对照，核对 metadata 任务数与两种口径。
本轮保留必要原始设备证据、复现脚本和一份本地输入；删除重复编译、profiler 中间产物与
冗余日志，不提交权重/张量大文件。最终字段名与文档整理不重复占卡。
改动 Python 文件的 Ruff、语法编译，三个复现脚本的 bash 语法，以及 Git 空白检查通过。

## 109. NZ 代表边界与动态补位图回放；后续集中优化性能版（2026-09-26）

设备算术沿用 `6ac7b9c7`，PyPTO `88297437`、Simpler `a54c05095`、PTOAS 0.66、
PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
本阶段均为单卡正式 `model.layers.2` 权重加固定 seed=1024 的合成输入/历史，S6、
atomic=0、Native deterministic level=1、HCCL_DETERMINISTIC=true；没有新增 16 卡任务。

### 109.1 边界与补位结果

任务 `task_20260926_151547_3935961537` completed/exit=0：
性能版 B1/H255 与精度版 B5/H32767，各自比较 mode=0/2。
四组同初态重复、同址 A/B/A 图、metadata 与保护区通过；同实现两种布局的
八类逻辑输出/状态精确相同。这证明所测边界的布局变化中性，不是跨实现精度通过。

新增 `--padding-graph`：Native builder 在图外更新同址 metadata，图内捕获 Native compact
producer 与完整 PTO 层；重放时 active B4→3→1→4，补位 seq_lens=0、slot=-1、页表=0，
positions/尾部 RoPE 保留旧值。独立 Native producer 按实际有效请求生成 compact oracle，
PTO 有效输出比较同实现满档前缀，非有效 cache/state 保持初态。
任务 `task_20260926_152243_50530218300` completed/exit=0：两版 B4/H4095、mode=2
全部重放的有效输出、全部 cache/state、compact 有效行、metadata 和保护区 PASS。
builder 的捕获输入地址保持不变。此证据不覆盖 Native 完整图、空 rank、全部档位，
也不替代真实模型逐步更新场景。

### 109.2 已观察到的差异与最新执行优先级

任务 `task_20260926_152534_53012716184` completed/exit=0，补测同输入 B1/H255 精度版。
两版 Native 八类基线状态精确相同；性能版 x_out 对 Native 为 max_abs=0.384277、
RMSE=0.026746，差异集中在前三个 token；精度版为 max_abs=0.015625、RMSE=0.001000。
性能版 ND/NZ 精确相同，Top-K 集合与 Native 相同而顺序不同。
目前没有判定该差异属于算术权衡还是功能问题，报告保持 MEASURED，未放行数值验收。

按用户随后明确的优先级：**性能优化先集中在性能版，尽可能对齐上游 pypto-lib；
形成稳定收益并完成输出 token 看护后，再回头补齐精度版性能**。
精度版保留当前对齐 Native 的算术方式；不要求每个候选同时修改两版。
当前精度暂时仅看护输出 token 一致，逐阶段、逐元素误差诊断后置，已有差异保留必要证据；
越界、漏写等功能问题仍须修复。下一步从第二层完整区间的现有热点推进性能候选，
不为每次参数调整启动 16 卡，也不重复展开本节差异诊断。

证据：
- [边界配置与任务](results/csa_baseline_20260926/nz_native_edges/manifest.json)
- [ND/NZ 八类状态比较](results/csa_baseline_20260926/nz_native_edges/comparison.json)
- [短上下文两版差异](results/csa_baseline_20260926/nz_native_edges/short_precision_diagnostic.json)
- [补位图配置与任务](results/csa_baseline_20260926/nz_native_padding/manifest.json)

保留报告全部字段、比较与复现脚本。已完成比较的重复张量、编译产物和冗余日志清理；
仅保留两份 B1 mode=2 状态和一份输入快照供后续恢复诊断，张量不提交。
改动 Python 的 Ruff、语法编译，三个 shell 脚本语法，以及 Git 空白检查通过；
最终整理只重建 CPU 差异报告，没有重复占卡。

## 110. 性能版 Q 展开对齐上游完整 K 投影，单卡完整区间降低 3.12%（2026-09-26）

基于 `82a37c8d`，参考 pypto-lib `216456332c2a74d89cca23b7824dab264ce34bff`。
PyPTO/Simpler/PTOAS/PTO-ISA 与第 109 节相同；本轮只修改性能版 Q 展开，精度版未修改。
所有计时均为单卡正式 `model.layers.2` 权重、合成输入/历史、seed=1024、B16/S6/H8192，
两侧 mode=2；PTO atomic=1、Native deterministic level=0、HCCL_DETERMINISTIC=false。
第二层复用 compact metadata；每侧 5 次预热、20 次完整图外设备事件采样。

### 110.1 保留改动与收益

性能版 Native NZ Q 展开从 N512、按 K128 分次 matmul_acc，改为上游的 N256、
完整 K1024 权重留在 L1，M64 的整行块与有效尾行分别 matmul。
Native 权重仍为 `[K,N]` NZ，直接复用原存储；不转置、不重排设备权重。
乘加仍为 INT8×INT8→INT32，未改变 QR/KV split=8/8、量化、softmax 或舍入策略。
ND 保留原有分块实现。此前关于旧布局和旧测量的冲突注释随修改删除。

任务 `task_20260926_155100_79968024243` completed/exit=0：

| 完整图区间 | p50（μs） | p95（μs） |
| --- | --- | --- |
| 原 PTO 基线 | 843.53 | 869.22 |
| 保留的 Q 展开候选 | 817.22 | 843.12 |
| 本轮相同 mode 的 Native | 921.57 | 926.48 |

PTO 中位数下降 3.12%。metadata、保护区、有限值及 Top-K 结构检查通过。
默认 atomic 的同初态浮点差异仍存在，不把这些功能检查写成逐元素精度通过。

DFX 任务 `task_20260926_155253_81161321182` completed/exit=0，复用已有 schema=2
输入，一份完整窗口、1131 条 worker 记录。完整调度区间 837.90→810.00 μs。
Q 展开最大单 worker 时间 57.98→49.24 μs，但其首末窗口受并发调度影响，
不能把整层收益全部归成该 kernel 的净时间，也不能累加各阶段窗口。
稀疏 QK/PV 与输出投影仍是后续关键路径候选。

受影响尾块任务 `task_20260926_155902_8638095291` completed/exit=0：
同一部署算术下 B1/H255 和 B40/H8192 均完成，有限值、Top-K 结构、metadata/保护区通过；
分别覆盖小于一个 M64 块和多个 M64 块加尾块。未追加详细精度诊断或完整档位测试。

### 110.2 未确认收益的候选

| 候选 | PTO p50 / p95（μs） | 处理 |
| --- | --- | --- |
| O-B 整段 K1024 权重常驻、N256 | 857.23 / 877.30 | 撤回 |
| O-A/B 按 token 数选择 32/96/128 行块 | 850.40 / 865.00 | 撤回 |
| 在 Q 展开候选上采用上游 QR/KV split=2/4 | 817.25 / 845.86 | 未确认额外收益，撤回 |
| 在 Q 展开候选上启用 WqB BYPASS | 823.08 / 828.66 | 中位数收益未确认，撤回 |

任务分别为 `task_20260926_154448_7552601248`、`task_20260926_154802_78344828780`、
`task_20260926_155455_8278903199`、`task_20260926_155651_8460714113`，全部 completed/exit=0。
跨轮次有波动；不把低于统计噪声的差距说成收益，不为未保留候选追加整模型测试。
被撤回项只留配置、原始采样和源码补丁，完整重复报告、构建与运行日志已删除。

证据：[配置与任务](results/csa_baseline_20260926/perf_qproj_upstream/manifest.json)、
[完整计时、泳道与候选汇总](results/csa_baseline_20260926/perf_qproj_upstream/comparison.json)。
保留候选报告的完整字段、原始 DFX/依赖/任务名称表和复现脚本，删除重复编译与日志；
第 109 节及本节报告将叶子记录压成单行，字段与数值不变，减少无效篇幅。
源码 Ruff、语法、shell 语法和 Git 空白检查通过。当前仍未达到 750 μs，
也没有宣称整模型性能通过；下一步只对保留候选进行正式 16 卡 token 看护。

## 111. 性能候选的 mode=2、默认 atomic 整模型 token 看护通过（2026-09-26）

性能源码固定为 `7eba45a3`，任务 `task_20260926_160622_9424198623`。
正式 75 分片权重、既有 H8192 bank、B16/TP1/DP-EP16、每请求输出 96 token，
两侧 mode=2、FULL_DECODE_ONLY、capture size=96；PTO performance、atomic=1、QR/KV split=8/8。
新进程不启用 `--deterministic`，Native 使用 level=0 默认值；明确设置 HCCL_DETERMINISTIC=false，
HCCL_OP_EXPANSION_MODE=AIV。任务期间冻结生产源码。

单卡先行：第 110 节完整区间收益，以及 B1/B40 的受影响功能检查均已完成。
当前只做逐 token 看护，同时保留运行自然产出的 DSpark 统计；不启动逐层或逐元素诊断。
任务 completed/exit=0。两侧 16 个 rank 均完整落盘，CPU 比较结果 PASS：
**24,576 个输出 token 完全一致，DSpark 总计与逐位置接受数一致，缺项和差异均为 0**。
各 rank 实际 mode=2；PTO 日志确认 performance、atomic=1、QR/KV split=8/8，
wq_a/wq_b/wo_a/wo_b 根布局全部 NZ。每个 rank 的 21 个目标 C4 层均捕获
`pto_tokens96` 路径，并观察到实际 aclgraph replay，避免把 Native 回退误当成 PTO 通过。

仅对这组配置声明 token/DSpark 看护通过；未追加逐元素或层误差诊断。
本轮 decode 的含加载/IO 总时长不作为整模型性能结论，750 μs 目标仍未完成。
后续继续集中优化性能版，精度版性能仍后置。

证据：[配置与任务](results/csa_baseline_20260926/model_b16h8192_nz2_perf_qproj/manifest.json)、
[全部 token 与 DSpark 对照](results/csa_baseline_20260926/model_b16h8192_nz2_perf_qproj/comparison.json)、
[实际配置、捕获和重放核对](results/csa_baseline_20260926/model_b16h8192_nz2_perf_qproj/execution_checks.json)。
保留两侧全部 rank 的 token/统计原始字段及精简执行证据，删除成功任务的重复进程/设备日志。

## 112. 性能版两项分块/流水候选撤回，补齐 mode=1 单卡测量（2026-09-26）

承接 `1c0517d9`，严格按性能版优先推进，精度版未改。沿用第 110 节的正式单层权重、
合成输入、B16/S6/H8192、atomic=1、level=0、HCCL=false；主计时复用第二个 CSA 层的
compact metadata，5 次预热、20 次图外事件采样。工具链保持 PyPTO `88297437`、
Simpler `a54c05095`、PTOAS 0.66、PTO-ISA `327cd586`。

| 单卡候选 | Native p50（μs） | PTO p50/p95（μs） | 决定 |
| --- | --- | --- | --- |
| mode=2，QK/PV 预发 2→1、槽 3→2 | 934.18 | 833.21 / 870.86 | 未测到收益，撤回 |
| mode=2，NZ Q 展开 M64→M96，N256/完整 K 不变 | 914.77 | 833.21 / 854.24 | 未测到收益，撤回 |
| 原保留版本，mode=1 | 939.49 | 872.30 / 906.92 | 保留另一档测量，主优化仍优先 mode=2 |

任务分别为 `task_20260926_162130_10627429948`、
`task_20260926_162631_108463729858`、`task_20260926_162949_110198329368`，
均 completed/exit=0，必要有限值、索引结构和保护区检查通过；不声明跨实现逐元素通过。
被撤回候选不再做尾块或 16 卡验证，仅保留实际样本、配置和相对 `1c0517d9` 的补丁，
其重复构建、输入快照和运行日志已删除。

仍保留第 110 节 mode=2 的 817.22 μs 版本及第 111 节 token/DSpark 通过结论。
mode=1 相对旧单卡基线没有确认收益；不能用当前单卡模式差异直接定最终整模型主口径。
下一步补齐真实模型测量，分别采无 profiler 完整步设备时间和各层 HC_pre→HC_post
设备首末区间，再按真实关键路径继续性能版优化。750 μs 目标仍未完成。

证据：[两项撤回记录](results/csa_baseline_20260926/perf_qproj_upstream/comparison.json)、
[mode=1 原始报告](results/csa_baseline_20260926/perf_qproj_upstream/mode1/report.json)。

## 113. 为性能版补齐整模型设备计时与严格窗口，先验证测量工具（2026-09-26）

审查发现旧 `offline_begin_profile` 无条件把每个调度步认作稳态，旧 `steady` 只读主机
`perf_counter`，且把调度 query token 数称为吞吐。这些口径不足以验收 B16/S6 或各层 750 μs。

本轮仅修改测量工具：按实际请求数与 query token 数筛选；profiler 启动后采连续窗口，
中途变档则报告不足；完整步使用图外 NPU Event，窗口内不逐步同步，收尾统一读时间戳，
检查时间戳确实更新。主机时间和调度 token 速率另列，不声称实际输出吞吐。
新增 `performance` 命令，在同一次模型加载中依次预热、无 profiler 测完整步、独立 Level0
采设备层区间，保留各轮 token、DSpark、实际档位与捕获路径。各层区间仍须解析首末任务，
不将并发 runtime/worker 耗时相加，也不将 profiler 的整步时间混入主性能采样。

CPU 回归 `test_csa_performance.py` 共 3 项通过，覆盖错误档位、窗口中途变档、独立设备事件。
随后单卡任务 `task_20260926_163524_112529814290` completed/exit=0：真实矩阵乘 ACL Graph
重放获得 20 个更新的设备时间戳，Level0 捕获 3 个完整步和 3 条设备任务，输出检查通过。
这只证明测量工具能工作，不是 CSA 性能或整模型结果。

证据：[单卡测量工具报告](results/csa_baseline_20260926/performance_measurement/report.json)、
[复现脚本](results/csa_baseline_20260926/performance_measurement/run.sh)。
下一步使用第 110/111 节保留且通过 token 看护的性能版，依次测两侧相同 mode=2/1。
