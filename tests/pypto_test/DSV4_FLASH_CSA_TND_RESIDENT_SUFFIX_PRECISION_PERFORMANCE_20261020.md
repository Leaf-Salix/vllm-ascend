# TND精度版：新SDK精度与常驻后缀性能实测（2026-10-20）

## 结论与版本

仍使用分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
本轮开始时HEAD为`dfeb16bd0d83e9e0c1f669966c0345cc2cb42517`，
正式运行源固定`48c58ce7c467dfce63d5cb5c41afd225488d6012`，未修改默认路由和已验收环境。

新SDK上的完整CSA在8K/B16/T96的八个观测边界对Native SK0逐位一致，但中位延迟1326.08 μs，
Native为936.84 μs，慢41.55%。换SDK没有取得可确认的性能收益。
常驻O-proj/HC-post候选只在独立诊断文件，尚未接入完整CSA，后续结果见下表。

## 测量口径

- 227单卡，真实DeepSeek-V4-Flash-0731-w8a8第4层权重，NZ2，TP1。
- 完整CSA使用合成hidden/history，history8191、B16、每请求S6、T96；范围HC-pre到HC-post。
- 后缀使用真实WO-A/B和加载后的通道scale，合成heads/residual/post/comb；仅O-proj与HC-post。
- 确定性level1及HCCL确定性开启，实际FULL raw graph replay；Native SK0、静态编译、DSA多流。
- 每臂12窗，每窗100回放，两臂顺序逐窗交替；保留全部样本，不开启DFX、依赖图或PMU。
- 表中延迟统计来自每窗100次回放的平均耗时；中位数是12个窗口均值的中位数，不是单次回放p50。
- 每项进程750秒、调度780秒硬上限；精度门禁先于计时，首差立即停止。

这不是DP16整模型测试。后缀比较对象为冻结PTO数学，不是直接运行Native后缀；
不能用此结果替代8K/B40、128K/B24的整模型192-token验收，也不能外推新SDK上所有形状均对齐Native。

## 完整CSA对Native

| 8K/B16，单位μs | 最小 | 中位数 | 平均 | 最大 | 精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| Native SK0 | 935.47 | 936.84 | 936.74 | 937.99 | 同输入控制 |
| 冻结TND精度CSA，新SDK | 1321.24 | 1326.08 | 1326.04 | 1330.21 | 八边界逐位一致 |

八边界为最终输出、TopK、raw KV、压缩KV、主state、Indexer两组cache和Indexer state。
A/B/A实时输入更新、输出毒化恢复、只读权重与cache guard通过，已确认PTO入口执行，不允许静默回退。
旧SDK同日上午控制为Native940.83 μs、CSA1333.73 μs。两次差距不足以证明SDK本身产生收益。
本轮未重复Native SK1：上一轮SK1/SK0已有边界差异，不能将两档混作同一精度金值。

## 常驻后缀候选设计与限制

候选SHA256为`a31a78f89e1ed71c007de2b692434cc0a0f1bc35626b22188418b3e715edd022`。
单个MIX使用24 AIC/48 AIV，保留冻结数学的Native K起点/次序、BF16落点、全8192列统一量化和HC-post左折叠。
原多任务按阶段拆开；本候选改为常驻逻辑worker、外部GM工作区及六阶段信号同步。
为解决新SDK编译器的L0C分支冲突，v4_wide_packet的stage2 pipeline改为顺序range，K迭代次序不变。
因此这是一个整体调度候选，性能差异不能单独归因于“SPMD”或信号开销；其中也包括流水方式改变。

后缀五边界是WO-A投影BF16、量化INT8、token scale FP32、WO-B输出BF16、HC-post最终输出。
先eager门禁，再FULL A/B/A、输出毒化、每窗逐位复核；输入/权重SHA只读检查、工作区/信号首尾guard、
每次回放generation推进均须通过。residual/output二维描述符与原三维Tensor共享pointer、offset和连续storage。
后缀没有请求metadata，不单独证明混合1～6行的TND入场支持。

## 常驻后缀结果

| 真token行数 | 原后缀 μs | 常驻候选 μs | 中位延迟变化 | 精度 |
| --- | ---: | ---: | ---: | --- |
| T18 | 271.65 | 390.69 | 慢43.82% | 五边界逐位一致，12窗复核通过 |
| T126 | 394.56 | 681.52 | 慢72.73% | 五边界逐位一致，12窗复核通过 |
| T240 | 466.27 | 638.38 | 慢36.91% | 五边界逐位一致，12窗复核通过 |

三档每一个配对窗口的常驻候选都更慢，回退不是仅凭一个中位数判断。
T240常驻候选最大窗口683.33 μs，仍按约定保留；其余窗口和全部原始样本均保留。
三档探针进程耗时约102/81/82秒；调度锁占用含环境准备，起止日志另存，均小于15分钟。
常驻候选不切换正式路由。没有取得稳定收益，因此本轮不启动新的8K/128K整模型验收。

## 运行历史

| 任务/检查 | 最终状态 | 说明 |
| --- | --- | --- |
| CPU reference v1 | 失败，无NPU | skip_ptoas=True缺少KernelArtifact要求的kernel_config |
| CPU reference v2 | 通过，无NPU | 两入口完整PTOAS/CCEC编译及Artifact加载 |
| task_20261020_170510_260331423946 | exit1 | Python前启动失败；无数学结果，原因未证明 |
| task_20261020_170742_277573825090 | exit0 | 显式16项S6；完整CSA新SDK对Native SK0 |
| task_20261020_171529_331384413694 | exit0 | 后缀T18，精度及性能完成 |
| task_20261020_171841_354695814903 | exit0 | 后缀T126，精度及性能完成 |
| task_20261020_172112_370870320045 | exit0 | 后缀T240，精度及性能完成 |

未删失败记录，也未将初始化/编译失败算作kernel精度失败。所有本轮设备任务已经结束。

## 工具链

PyPTO `2dfa7f120ba5e577566046d920845aa241bfa7bb`，Simpler `57e499238fd4ee3548947ad860be456aaeeb16b5`。
PTO-ISA `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`。
实际PTOAS为隔离0.66 wheel，SHA256 `d121b5b5db9a23a98eb3c0acc74d0abaef8aaa90e4df0385e79feb67636ef997`；
wheel的Git源提交未证明，不能等同已准备的PTOAS参考源。
Python3.12.14、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
虽然提交命令有`--ptoas 0.63`，脚本明确覆盖PTOAS_ROOT为新隔离toolchain-wheel，结果记录实际导入与DSO SHA。

## 未完成范围

新常驻后缀未接入完整CSA，未新增8K/B40或128K/B24整模型token资格。
nalinaly完整动态SPMD目前只有离线编译资格，没有本轮同环境设备性能值；用户提供的性能截图仅作为目标。

## 证据与复现

[原始证据索引](evidence_tnd_precision/resident_suffix_20261020/README.md)保存完整JSON、失败历史、逐项SHA和原样测试脚本。
[复现步骤](evidence_tnd_precision/resident_suffix_20261020/REPRODUCTION.md)区分完整CSA与后缀资格。
