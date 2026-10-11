# MIX 成员 Scalar 提前执行：功能与128K/B16性能

日期：2026-10-11。**功能通过，新增MIX标记尚未取得性能收益。**

## 当前结果

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通路径 | 972.12 | 989.03 | 1014.58 |
| 原 early_start | 951.24 | 981.37 | 1009.28 |
| early_start + MIX | 962.82 | 992.01 | 1014.60 |

同SDK、单卡1，CANN9.2/TMR/NZ2/atomic=0/deterministic=0，关闭诊断采集。
第2层真实权重与合成128K历史，B16×验6；区间含HC_pre+norm+CSA+HC_post。
每条路径5次预热、30次轮换顺序的单次图重放，缓存恢复在计时外。
任务`task_20261011_182414_219668930373`退出0，设备已释放。
相对原early_start，MIX均值增加10.64μs（1.08%），max增加5.32μs。
两候选的9项输出/缓存/状态，以及改变输入后的图重放均与普通路径逐bit一致；
保护区通过。完整90条原始样本见[csa_results.json](csa_results.json)。

## 功能证据及边界

- 连续MIX任务两轮改变输入：24/24 AIC和48/48 AIV均在前驱整组完成前进入Scalar。
  AIV两轮均48/48形成同核交接；尾部等待超过5个设备tick的分别2/4个。
  AIC这组用例走空闲核前缀，不能用它声称24个本核Cube尾部均重叠。
- 数值及保护区通过；完整时序见[mixed_chain_results.json](mixed_chain_results.json)。
- 第一轮设备任务`task_20261011_181145_167147611826`有7项通过、2项失败。
  基础MIX测试误用了已废弃的JIT RunConfig调用，改用ChipWorker；旧sync_start
  断言要求无依赖的全部X block先于B完成，比原调度语义更强，已改为所选物理核
  的实际前驱集合。原失败没有保存完整数组，不能仅凭缩略报错确定当轮具体时序。
  后续测试已增加断言前完整时序保存。两失败项在
  `task_20261011_181414_17692867746`定向复测通过，原始XML均保留。
- 原CPU能力检查11项通过，PyPTO编排3项及参数加载2项通过；PTOAS双向pipe
  FileCheck及非法前缀拒绝检查通过。源码、编译结果与设备运行分开记录。
- CSA共73个编译成员，73个带Wait、71个带Set；其中12个MIX成员全部带两者。
  这是编译覆盖数，不是128K/B16实际交接次数。8K分支没有在本轮实测。

## 实现与复现

版本见[versions.json](versions.json)，三仓补丁与CSA六条局部语句补丁随目录保存。
Simpler按AIC/AIV0/AIV1传递能力，保留原选核、队列、ACK和组完成统计。
PyPTO让自动参数加载和pipe初始化位于Wait之后；PTOAS仅为runtime-owned
PIPE_ALL扩展MIX操作，旧event7合同保持受限。

使用私有环境`../.worktrees/scalar-overlap-demo-20261009/env.sh`及目录内的
运行脚本；所有设备命令仍通过`task-submit --device auto --device-num 1`执行。
先CPU编译三条路径，再运行`run_csa_mixed_prefix_128k_b16.sh`。

用户确认继续手工标记`wait_pre_task_end()`，不做自动识别；chip swimlane、PMU
及参数dump保留串行回退。当前尚未消除单次探测错过后到任务的窗口，
也未证明CSA各任务的实际重叠命中率或稳定净收益。下一步先去除gated交接
额外放到AICPU的参数准备，再判定性能；这不是MIX功能失败的结论。
