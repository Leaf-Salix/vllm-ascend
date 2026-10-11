# 32位派发代次通知：减少尾部回调的准备开销

2026-10-12，过程077。**保留为私有TMR优化，公共CSA默认路径未切换。**
仍用手工局部Wait/Set；chip swimlane、PMU和参数dump回退保持，不改原派发、ACK、核选择和early-resolve。

## 机制与安全边界

原通知先失效、读取旧GM通知值，再选相反值填充片上scratch并写回。
新通知在派发ID尚未复用时使用`reg_task_id | 0x80000000`，省掉这次旧值读取。
AIC使用PTO-ISA已有的uint32 L1常数填充，AIV填充16个uint32；占用和写回仍为64B。
有效ID的高位编码与冷初始化1、旧协议的0/全1分离；完整流水汇合、通知轮询及最后的PIPE_ALL不变。
通知值和填充宽度同时变化，测量证明的是整个候选的收益。

第一次寄存器任务ID回绕时，Simpler在CoreExecState原padding中的sticky位记录“ID已复用”。
编译通知随后恢复读取旧值并翻转；该位跨后续uint32回绕保持，冷初始化清除。
它通过现有context字段传递，直接通知分支仍只收到原scratch地址，不把标志当作地址。
CoreExecState/DispatchPayload仍为64/576B，任务代次和调度决策不改变。

没有采用“本核缓存一个翻转相位”的设计：部分回调可能直接阻塞而不写通知，
仅凭回调被调用就翻转相位会失去与真实通知值的对应。代次编码不需要维护这第二份相位状态。

回调仍可能因没有通知事件而走阻塞回退；此候选没有消除那段观察窗口限制。
一般只读GM Scalar前缀的写者/别名证明也仍未补齐。

## 单卡性能

CANN9.2/TMR，128K/B16第二层HC_pre+norm+CSA+HC_post，NZ2、atomic0、deterministic0，profiler关闭。
三路用同一SDK、权重和算术；每轮各5次预热、30次轮换单次NPUGraph重放，reset不计时。

| 轮次/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 首轮普通PTO | 969.40 | 986.41 | 1009.14 |
| 首轮原通知 | 971.48 | 991.01 | 1007.64 |
| 首轮代次通知 | 958.44 | 984.58 | 1012.48 |
| 确认普通PTO | 969.34 | 985.77 | 1003.98 |
| 确认原通知 | 968.84 | 985.04 | 1000.94 |
| 确认代次通知 | 963.62 | 981.38 | 995.38 |

对原通知，两轮mean减少6.43/3.66μs，配对中位差−7.24/−2.32μs，21/30和20/30配对更快，因此保留。
max变化+4.84/−5.56μs，没有稳定长尾收益证据。
对普通PTO，mean减少1.82/4.39μs，但首轮配对中位差为+0.48μs、14/30更快；
确认轮为−5.17μs、19/30更快。仅据此记录本档小幅均值优势，不泛化为所有场景或整网加速。

[首轮90条样本](round1/results.json)、[确认轮90条样本](confirm/results.json)、[配对计算](paired_comparison.json)。
两轮三路9项输出/缓存状态及改变hidden输入后的重放逐bit一致，每路31项保护区通过。

## 功能与编译检查

- [17项CPU调度器测试](notification_id_cpu_20261012.xml)：含首次ID回绕、之后的整数回绕、直接通知地址隔离及冷初始化复位。
- [23项后端UT](notification_id_backend_20261012.log)：通知宽度、回绕分支、事件分配/中转/阻塞合同。
- AIC/AIV的[正常代次通知](functional/generation_results.json)与[强制回绕翻转](functional/wrapped_results.json)，均检查两次输入、图重放、Scalar结果、输出/trace保护区和完整尾部等待。
- [MIX两项设备检查](functional/mix.xml)通过，包含FIFO、各物理成员局部边界和sync_start。

强制回绕用编译分支注入验证设备实现；CPU测试单独验证真实发布条件，没有运行数十亿个task。
小用例trace用于功能证明，`repeats=0`，不与未插桩CSA计时混用。
两路early-start的程序IR完全相同，各73个成员通过Wait前流水检查：
[原通知检查](compiler/original_notice_prefix_audit.json)、[新通知检查](compiler/dispatch_notice_prefix_audit.json)。
[回调清单](compiler/notification_manifest.json)共71个函数，不代表动态调用次数；
[AIC回调](compiler/aic_notification.cpp)、[AIV回调](compiler/aiv_notification.cpp)保留为生成代码证据。

任务`task_20261012_043856_19092799607`和`task_20261012_044240_205835017095`均卡0退出0并释放。
没有新增Native、七档、16卡、token或DSpark结果。

## 版本与复现

保留版本：Simpler `1d09ecd34`、PyPTO `3cf735fe2`、私有vLLM驱动 `ebbb5fd0`；PTOAS/PTO-ISA未改。
测试发生在PyPTO提交前，使用`ef76af347`加本次同内容patch；SDK绑定、TMR和Torch适配器都已重建并核对Simpler版本。
[完整版本](versions.json)、[Simpler修改](simpler.patch)、[PyPTO修改](pypto.patch)。

将[构建脚本](repro/build_notification_id_sdk_20261012.sh)、[首轮脚本](repro/run_notification_id_20261012.sh)
和[确认脚本](repro/run_notification_id_confirm_20261012.sh)放在对应私有环境的`artifacts/`下；父目录含`env.sh`及三个私有仓。
设备脚本经`task-submit --device auto --run 'bash <script>'`执行，使用队列分配的`TASK_DEVICE`。
原通知生成器显式冻结在[参考代码](repro/tmr_tail_notification_control.py)，对照驱动只替换通知回调生成方式。
