# AIV 通知改用普通 mask：功能通过，CSA 回退后撤下

2026-10-12，过程081。候选没有保留，PyPTO已恢复到`f6edfc1fd`的count模式填充。
Simpler、PTOAS、运行时ABI和公共CSA默认均未修改。

## 改动和比较范围

通知每次只填充64字节，即16个uint32。原实现切到count模式，以数量16执行`vector_dup`，随后恢复普通模式和全mask。
候选先设置普通模式及低16位mask，执行一次repeat，再恢复全mask，少一次模式设置调用：

```cpp
// 原实现
set_mask_count();
set_vector_mask(0, 16);
vector_dup(scratch, expected, 0, 1, 1, 8, 8);
set_mask_norm();
set_vector_mask(~uint64_t{0}, ~uint64_t{0});

// 已撤下的候选
set_mask_norm();
set_vector_mask(0, 0xffff);
vector_dup(scratch, expected, 1, 1, 1, 8, 8);
set_vector_mask(~uint64_t{0}, ~uint64_t{0});
```

scratch大小、32位代次值及回绕回退、64B DMA、全部流水汇合、轮询和最终完整等待不变。
原派发、ACK、核选择、early-resolve和手工局部Wait不改；chip swimlane/PMU/参数dump串行回退保留。
CSA算术相同，73个成员中46个AIV通知回调有上述变化，AIC生成代码不变。
[逐成员差异记录](compiler/source_comparison.json)的一一源码比较忽略编译器内联变量编号差异，
[73个前缀检查](compiler/prefix_audit.json)均无Wait前流水操作。
[原回调实例](compiler/example_counted.cpp)、[候选实例](compiler/example_normal.cpp)、[完整候选补丁](candidate.patch)。

## 单卡功能

相关[26项后端单测](diagnostics/backend.xml)通过；首次仅一项仍匹配旧count-mask字符串，
将断言改为候选的16位mask、一次repeat及无count-mode后通过，通知值、回绕和流水检查保持。
[首次结果](diagnostics/backend_initial.xml)和[最终日志](diagnostics/backend.log)保留。

任务`task_20261012_062316_161726716713`，卡0退出0并释放。
AIV两次输入分别触发48、39次通知；AIC对照为22、24次。
两种核型的两次输入和图重放均检查输出、Scalar依赖、保护区与完整尾部时序，并通过。
此诊断不计性能：[结果](diagnostics/results.json)、[AIV两次原始trace](diagnostics/trace_False_1_late_0.json)、
[第二次trace](diagnostics/trace_False_1_late_1.json)、[AIV重放](diagnostics/replay_False_1_late.json)。

## 128K/B16完整CSA性能

CANN9.2/TMR、卡0、第2层HC_pre+norm+CSA+HC_post，NZ2、atomic0、deterministic0，关闭profiler。
三路各预热5次、30次轮换单次NPUGraph重放；reset在外部NPU Event计时区间之外。
普通PTO和当前count模式复用过程079同SDK产物，候选独立编译。

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通PTO，无early-start | 964.98 | 987.51 | 1008.02 |
| 当前early-start，count模式 | 957.20 | 984.49 | 1016.82 |
| 候选early-start，普通mask | 989.30 | 1008.89 | 1028.84 |

候选对当前实现：min多32.10μs、mean多24.40μs（+2.48%）、max多12.02μs。
配对中位差+26.27μs，仅6/30配对更快；因此撤下，没有为明显回退再安排确认轮或七档测试。
少一次源码模式切换不等于实际更快。本轮未测核内指令耗时，不能将回退归因于某条指令或认定Scalar-bound。
当前保留实现本轮mean比普通PTO少3.03μs，但max多8.80μs，不据这一轮宣称稳定长尾或整网收益。

三路各31项保护区通过；两条early-start相对普通PTO的9项输出/cache状态、换hidden输入后重放均逐bit一致。
[完整90条样本及功能结果](timing/results.json)、[配对差值](paired_comparison.json)、[运行日志](timing/run.log)。
任务`task_20261012_062342_16358076572`卡0退出0并释放；无新Native、token、七档或16卡结果。

## 版本、回退与复现

测试基于私有PyPTO `f6edfc1fd`加候选补丁，Simpler `1d09ecd34`、PTOAS `4df4627e`、
私有vLLM驱动基底`7f9f8213`。[完整版本](versions.json)。测试后只撤回本轮PyPTO两文件改动，
候选驱动仅归档在本目录，不保留为当前运行入口；不改其他会话的文件。
Python后端恢复为已有验证版本，因此不重复设备测试。

在相同私有环境内应用`candidate.patch`，将`repro/`的两个Python驱动放回私有vLLM的`tests/pypto_test/`，
两个shell脚本放到私有环境`artifacts/`。从私有vLLM仓source父目录`env.sh`，设置NZ2/atomic0，然后编译：

```bash
python tests/pypto_test/tmr_csa_normal_notice_compile.py \
  --output ../artifacts/csa_normal_notice_128k_b16_20261012/compiled \
  --controls ../artifacts/csa_untraced_notice_128k_b16_20261012/compiled
```

诊断从私有PyPTO目录运行`task_boundary_runtime_poll_probe.py --compile --compiled <diagnostic-dir>/compiled`。
设备统一通过`task-submit --device auto --device-num 1 --run 'bash <script>'`排队，脚本使用`TASK_DEVICE`。
依赖现有冻结算子包、同SDK对照产物和第2层权重；本归档不是全新clone可直接执行的发布包。
后续继续处理安全只读GM前缀的依赖/实际别名依据及交接开销，不重复本次普通mask候选。
