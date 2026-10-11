# TMR 已消费边界省去指针清零：128K/B16 小幅收益，保留

2026-10-12，过程083。Wait之后不再将`predecessor_context`和`wait_predecessor_fn`写零，
仍在FIN之前刷新控制缓存行。候选128K/B16均值979.47μs，比前后原early-start实现分别少3.44/6.17μs，
min和max也均低于两次原实现，保留到私有Simpler `7043f8c1b`。
本次只证明指定档位的改善，不外推为七档或整模型收益。

## 修改及正确性依据

只修改`wait_scalar_prefix_predecessor()`中的两次volatile GM写零及相邻注释。
显式局部Wait的tagged入口由编译器保证只有一个有效Wait；旧入口在消费后清除`local_predecessor_pending`，
后续不会再次调用等待回调。因此两个指针只在进入/消费边界时使用，消费后的旧值不再读取。
原AICPU派发准备仍在发布新代次之前初始化两个字段。该初始化不可省略。

保留以下原流程：

- 前驱所有本核流水的完整等待。
- 当前控制缓存行的DCCI写回，仍在FIN之前；防止晚到写回覆盖CPU的新派发数据。
- 旧入口的pending标志清除、全部fan-in/cohort等待、延迟ACK和退出检查。
- 原派发、核选择、队列和`allow_early_resolve`，以及手工局部Wait/Set接口。
- chip swimlane、PMU、参数dump开启时的串行回退。

这不是跳过Wait，也没有扩大不安全输入读取范围。
[保留的Simpler提交](simpler.patch)只有1文件+4/−5行；
[PyPTO提交](pypto.patch)只将私有SDK引用更新到该版本，1文件+1/−1行。
PTOAS、PTO-ISA、CSA算子算术与公共默认入口不改。

## 单卡功能

同一张卡上先运行既有AIC/AIV/MIX三个定向用例，全部通过。
纯AIC两次输入各24对、纯AIV各48对通过同核交接与完整尾部等待时序检查，输出、Scalar结果及保护区通过。
MIX两次均有24个AIC和48个AIV成员在前驱组结束前进入前缀，保留FIFO与sync_start。
组级提前入口不代表已经证明每个成员的原始计算重叠。

[AIC原始trace](diagnostics/local_aic_branch.json)、[AIV原始trace](diagnostics/local_aiv_branch.json)、
[MIX原始trace](diagnostics/mixed_chain.json)、[JUnit结果](diagnostics/results.xml)、[日志](diagnostics/run.log)。

随后每次CSA进程中的early-start与普通PTO比较：9项输出/cache状态逐bit一致，
各路径31项保护区通过；图捕获、5次预热以及改变hidden输入后的图重放均通过。
普通路径没有局部Wait，因而不会调用被修改的等待回调；它也作为同进程性能参照。

## 128K/B16性能

单卡0、CANN9.2/TMR、第2层HC_pre+norm+CSA+HC_post，NZ2、atomic0、deterministic0，无profiler。
保持相同算子二进制和算术，仅切换运行时。顺序为原运行时→候选→原运行时，三个独立进程共享一次卡锁。
每个进程中两路径各5次预热、30次轮换单次NPUGraph重放；reset位于NPU Event计时区间外。

| 运行时/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 前测原运行时，普通PTO | 973.50 | 992.29 | 1006.26 |
| 前测原运行时，early-start | 964.50 | 982.91 | 1009.78 |
| 候选运行时，普通PTO | 972.64 | 997.32 | 1018.92 |
| 候选运行时，early-start | 957.22 | 979.47 | 1006.66 |
| 后测原运行时，普通PTO | 977.14 | 997.98 | 1034.20 |
| 后测原运行时，early-start | 963.78 | 985.64 | 1007.34 |

候选early-start对前/后原实现，min少7.28/6.56μs、mean少3.44/6.17μs、max少3.12/0.68μs。
相对前后原实现均值的中点少4.81μs（约0.49%）。
同进程early-start−普通PTO的配对均值，前/候选/后分别为−9.37/−17.85/−12.34μs；
配对中位数为−9.56/−19.59/−8.98μs，较快配对22/27/20，共各30对。
普通PTO自身均值前后漂移5.69μs，不能忽略；候选对两次原实现和同进程参照的方向均改善。
不把0.68μs的max差单独表述为已证明稳定长尾优化，也不反复补采筛选更好的成绩。

[差值汇总](comparison.json)、[前测](timing/control_before/results.json)、
[候选](timing/candidate/results.json)、[后测](timing/control_after/results.json)。
`comparison.json`给汇总，180条原始采样位于三个`results.json`的`samples`字段。
任务`task_20261012_065801_290987017914`退出0并释放卡；没有新增Native、七档、16卡或token测试。

## 版本、构建与复现

测量时使用PyPTO `f6edfc1fd`、Simpler `1d09ecd34`为基底；候选运行时在独立目录应用
[实际测量补丁](measured_candidate.patch)，构建后恢复源文件，没有替换两个对照的运行时。
保留版本只是把同一可执行改动提交到Simpler `7043f8c1b`，并澄清一条生命周期注释；
PyPTO `2b10dfa65`更新SDK引用。保留实现的行为与已测候选相同。

私有binding、正常运行时与torch桥接已重建，[revision检查](retained/sdk_revision.log)一致；
普通/early-start CSA产物也已按新SDK重编，73个成员[前缀检查](retained/prefix_audit.json)无Wait前流水指令。
未因版本引用和注释变化重复占卡。旧SDK产物不能直接用于新SDK，后续使用
私有环境的`artifacts/csa_consumed_context_128k_b16_20261012/retained/compiled/{plain_score,early_score}`。
[完整版本与路径](versions.json)。

复现原A/B/A需要上述测量基底、冻结算子包、同SDK的过程079两路CSA产物及既有Native fixture。
把[实验驱动](repro/tmr_csa_consumed_context_demo.py)放回私有vLLM的`tests/pypto_test/`，
在该仓source父目录`env.sh`后，先用`--runtime-root <输出>/runtime --build`做CPU构建。
将[设备脚本](repro/run_consumed_context_20261012.sh)放回私有环境`artifacts/`，
通过`task-submit --device auto --device-num 1 --run 'bash <脚本>'`执行，使用队列分配的`TASK_DEVICE`。
脚本基于测量基底，不能在已经省去清零的源码上重复应用补丁。
[保留版本构建脚本](repro/build_consumed_context_sdk_20261012.sh)用于更新私有SDK及CSA产物，不占NPU。
实验入口只归档，不留作当前算子路径。

后续继续处理一般只读GM前缀的依赖/实际别名依据、更深控制流和事件资源耗尽后的观察窗口。
此次减少交接开销不代表所有安全前缀已支持，目标仍继续进行。
