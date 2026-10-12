# 过程088：gated 参数直读，保留到私有实现

2026-10-12。Simpler `3d0b9c820`、PyPTO `999c1f59f` 已保留；公共 CSA 默认不变，未 push。
PTOAS `fdebb38b`、PTO-ISA `327cd5869` 不变。

## 改动

原路径由 AICore 遍历 Tensor/Scalar 参数写入 args，wrapper 随后再读。
新入口直接读取 TaskPayload 的不可变描述符和 Scalar 原始位值。
既有能力字节高位声明兼容性，入口指针 bit 2 选择参数来源；bit 0 的延迟 Wait、
bit 1 的尾部回调保持独立。低三位成员掩码仍区分 AIC/AIV0/AIV1。

不扩展布局、不修改算子算法，不把参数能力当作提前执行资格。
原派发、核选择、ACK、early-resolve、局部完整尾部及 fan-in/cohort 等待保持。
旧入口和 chip swimlane/PMU/参数 dump 继续填充原数组，ready 路径使用原数组。
作者仍用局部 Wait/Set，当前不做边界自动识别。

## 单卡夹测

卡 0，CANN 9.2、TMR、128K/B16，第 2 层 HC_pre＋norm＋CSA＋HC_post。
NZ=2、atomic_add=0、deterministic=0，无 profiler。
原→候选→原三个进程共用一次卡锁；每进程两路各 5 次预热、30 次轮换单次图重放，reset 不计时。

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 前测原实现，普通 PTO | 968.82 | 990.00 | 1014.54 |
| 前测原实现，early-start | 958.52 | 990.58 | 1011.54 |
| 候选，普通 PTO | 975.74 | 995.70 | 1017.28 |
| 候选，early-start | 958.42 | 981.94 | 1000.46 |
| 后测原实现，普通 PTO | 975.72 | 993.89 | 1029.50 |
| 后测原实现，early-start | 966.72 | 993.16 | 1029.84 |

候选 early-start 对前/后 mean 少 8.63/11.22 μs，max 少 11.08/29.38 μs；
min 少 0.10/8.30 μs。相对控制均值中点少 9.93 μs，控制前后漂移 2.59 μs，因此保留。
普通 PTO mean 却多 5.70/1.81 μs，不能宣称未标记路径没有开销，也不能把全部配对差归于收益。
同进程 early−plain 配对均值为 +0.58/−13.76/−0.73 μs，12/24/14 轮 early 更快。
这些是单卡单档结果，不能与不同版本各轮收益相加，也未外推七档或整网。

三个进程两路各 31 项保护区通过；9 项状态及更换 hidden 输入后的重放与普通 PTO 逐 bit 一致。
这不是 Native/token/16 卡验收。

## 功能与正式接入

候选先通过 AIC/AIV/MIX 三项功能。两次输入 AIC 各 24/24、AIV 各 48/48 同核完整尾部交接；
MIX 每次 24 AIC/48 AIV 在前驱组结束前进入前缀。输出、Scalar、保护区、等待断言通过。
这些 trace 包含完整尾部/通知等待，不能冒充逐成员原始计算重叠或 5/50 μs 功能验收。

正式接入的参数及入口 UT 40 项、成员能力/early-resolve UT 4 项通过。
一项旧 UT 误要求生成代码中早已不存在的注释；去掉注释文本依赖后，
仍检查原子状态重置的编译保护、次数和先后顺序。

私有 binding、正常运行时、torch 桥接和两路 CSA 已重建，SDK revision 均为 `3d0b9c820`。
整理后的 73 个 kernel 原始指令字节，以及 AICore 执行器的 ELF 可执行代码段，
与已测隔离候选一致。该一次性比较不使用 hash，避免因代码整理重复占卡。
没有新增正式接入后的性能采样。

## 尝试过程与复现

- 首次 CPU 编译漏设 NZ=2。去除 SSA 注释后发现 6 个投影主体变化，尚未上卡；
  修正配置后 73 个主体与冻结对照一致。两次编译及差异清单保留。
- 首次任务 `task_20261012_084006_255465628038` 因 source 环境前未进入 Git 目录退出 128，
  没有进入功能测试；补齐 cwd 后执行 `task_20261012_084046_25800015566`，退出 0，卡已释放。
- 系统 clang-format 12 不支持仓库配置，私有 venv 安装 20.1.8 后仅格式化改动行；
  不修改公共 Python 环境。

[全部 180 条采样及统计](analyze.py)、[比较结果](comparison.json)、
[隔离运行时补丁](runtime/candidate.patch)、[正式补丁](retained/simpler.patch)、
[PyPTO 补丁](retained/pypto.patch)、[版本](versions.json)。
`diagnostics/` 保存原始功能 trace，`retained/` 保存 SDK 与指令一致性结果。

复现隔离候选需使用 baseline Simpler `7043f8c1b`、PyPTO `a330b3eb7` 和匹配 SDK，
不能在已经接入源参数的生成器上再次套实验 hook。
将 [实验入口](repro/tmr_csa_source_args_demo.py) 放在对应私有 vLLM 的 `tests/pypto_test/`，
先在 Git checkout 内 source 私有环境，设置 `VLLM_ASCEND_ENABLE_NZ=2`、
`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`，执行 `build`、`compile` 子命令，再由 task-submit
执行 [单卡脚本](repro/run.sh)。[正式版本构建脚本](repro/build_retained.sh) 只用于保留版本。
不归档权重、二进制或运行时缓存。

## 后续验收口径

用户明确收敛为三项：无 Wait/Set 的普通脚本相对引入功能前有无回退；
针对性场景证明同核后继 Scalar 与前驱真实 Vector/Cube 计算重叠；
实测单 incore task 约 5 μs、50 μs 时分别比较链路 min/mean/max。
固定任务数量和 Scalar 工作量，分开功能有效与性能收益。
当前 CSA 数据不能代替这些检查，接下来不继续铺 CSA 微优化实验。
