# 过程067：修复运行时通知的AIC/AIV弱符号混用

2026-10-12。私有A3/TMR、单卡。运行时直接DMA通知的核类型路由已修复，AIC/AIV及MIX功能通过；
当前直接通知实验尚无CSA性能收益，默认继续使用现行编译回调。
Simpler `e6b6ca49a`，PyPTO `d7e4b2578`，私有vLLM `91b6d0bb`，PTOAS `0c21f984`未改。
[完整版本与增量](versions.json)。公共环境及默认CSA算子未改。

## 旧超时的具体问题

过程058的直接通知在AIV上卡住，通知搬到HBM后仍失败，当时未确定编译上下文差异。
本轮发现`aicore_execute`和`aicore_execute_kernel`都以同名weak符号分别编译AIC/AIV版本；
实际链接顺序保留AIC定义，两个外层kernel入口均引用它。原执行器本身主要执行Scalar控制，
但旧`poll_late_handoff`在其中使用`#if __DAV_CUBE__`选择DMA和事件指令，
使AIV也可能进入AIC编译出的Cube指令分支。这是实现错误，不是AIV不支持运行时直接通知。

用旧`3ea4375fe` executor源码和当前同布局头文件重建原链接，交叉引用明确显示：

```text
aicore_kernel_mode_0_mix_aic ─┐
                            ├→ aicore_execute_kernel（AIC weak定义）
aicore_kernel_mode_0_mix_aiv ─┘    → AIC版本 poll_late_handoff
```

该阶段至当前的CMake链接顺序与kernel-mode外层入口没有变动。
[旧链接引用](link/former_kernel_mode.map.excerpt.txt)、[原executor源码](link/former_executor.cpp)、
[重建命令](link/commands.json)。这是源码重建证据，并非声称重新执行了旧失败设备二进制。

修复为两个名称不同的强符号`tmr_poll_local_tail_aic/aiv`，由真实worker的`CoreType`选择。
保留原弱符号Scalar执行器，架构相关指令只放进各自强符号函数；
[修复后链接引用](link/fixed_kernel_mode.map.excerpt.txt)显示两者分别来自对应目标文件，未被合并。
编译器生成task的回调此前能工作，是因为其二进制由具体task类型分别选择，不走这处弱符号合并。

## 当前实验边界

- 只恢复旧实验协议`RuntimePollable=6`的直接通知。默认`RuntimeCompiledPollClean=8`仍保留。
- scratch仍由编译器正式预留；实验hook在局部Set写入scratch地址和协议类型。
  通知使用64B DMA、完整流水汇合与event7；hook只接受已确认预留event7的函数。
  显式非7事件的默认编译回调能力没有撤回。
- 原首次探测失败才观察后到派发；原派发、ACK、核选择、early-resolve、完整fanin与cohort等待不变。
  后继仍使用局部Wait，MIX保持整组完成，DFX仍串行回退；没有自动识别前缀。
- 此实验还含Set的GM写入及旧协议的保守控制行flush，尚未迁移过程059已经验证的元数据传递方式。
  因此它是修复后的直接通知对照，不能称为最终最小开销实现。
- 显式诊断trace增加核类型位，未开启trace时不写这些位；新判断只在旧实验协议被选择时进入直接通知。

## 单卡功能结果

任务`task_20261012_003341_10316804416`，卡0，退出0并释放卡。先做AIC/AIV、再做MIX、最后做CSA。
纯核小用例loops=4、额外Scalar循环=0，两组输入及图重放通过输出、真实fanin、保护区和完成顺序检查。

| 核类型 | 输入0后到交接 | 输入1后到交接 | 重放后到交接 | 三次通知总数/正确核型 |
| --- | ---: | ---: | ---: | ---: |
| AIV | 15 | 0 | 5 | 141 / 141 |
| AIC | 13 | 23 | 19 | 69 / 69 |

后到交接关联同核、同generation消费者，核类型由实际执行的通知函数写入并显式断言。
原trace word28（通用尾记录word4）低8位7仍为完成标记；bit8为通知、bit9为后到交接，
bit12/13分别标识AIC/AIV函数。没有用尾等待时长充当原计算独立重叠时长。
[小用例完整结果](small/results.json)、[逐次核型复核](small/engine_route_counts.json)。
初次运行的通用打印器对字符串取长度，因此日志`verified_runtime_engine: 3`只是`aic/aiv`的字符数；
核类型断言实际执行过。事后仅改成已验证block列表方便计数，并离线复核原trace，没有重跑设备。

MIX两项检查通过，含跨核FIFO、两侧局部边界和sync_start；两轮各24个AIC、48个AIV成员提前进入安全前缀。
producer AIC及consumer AIC/AIV的实际通知核型全部匹配；producer AIV走立即交接时没有发通知，
不能将通知数0误读为缺少MIX支持。[MIX完整记录](mix/mixed_chain_results.json)、
[核类型检查](mix/engine_route_counts.json)、[pytest结果](mix/junit.xml)。

## 同SDK的128K/B16 CSA

三个路径均用新SDK重新编译，未复用旧revision的设备二进制。第二层真实权重、128K合成历史、
B16×verify6，区间为HC_pre+norm+CSA+HC_post。CANN9.2/TMR/NZ2/atomic0、deterministic level0，
关闭诊断；预热5次后各30次轮换单次NPUGraph重放，NPU event计时，reset在区间外。单位μs：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通PTO baseline | 975.70 | 991.13 | 1019.60 |
| 现行编译回调 mixed | 960.40 | 986.42 | 1007.22 |
| 直接通知实验 direct | 965.26 | 994.61 | 1106.76 |

direct比mixed mean多8.18μs（约0.83%）、max多99.54μs；11/30配对更快，中位差+6.00μs。
比普通PTO mean多3.48μs，中位差+4.20μs。候选没有净收益，且最大值回退，默认不切换。
Set写入和flush是已知额外操作，但本轮没有把它们与调用组织、调度波动分离计时，
不能把8.18μs或1106.76μs长尾全部归因于某一项。

9项输出/缓存/状态及换输入重放逐bit一致，每路31项保护区通过。
[全部90条样本](timing/samples.csv)、[完整结果](timing/results.json)、[配对差值](timing/paired_comparison.json)。
这是三种PTO实现之间的比较，没有新增Native、七档或16卡验收，也未证明完整CSA稳定净加速。

## 构建修正、保留项与复现

构建时首先发现Python KernelABI仍留旧revision，而binding/torch adapter已经更新；
同步完整SDK提交号后严格检查通过，没有放松版本校验或让混合版本上卡。
另一次CPU预编译因once对照已去掉helper，外层实验hook仍查找它而失败；
修正为检查实际生成代码后重新编译通过。两次失败日志保留在[build](build/)。

保留核类型入口修复及当前默认编译回调。下一步将直接通知的scratch/事件合同接入原任务元数据，
消除实验Set写入与保守flush后再评估性能；保持原派发及所有已支持的前缀机会，避免重做已排除的小入口候选。

在已有冻结CSA包的私有环境中恢复[versions](versions.json)，不要以runtime symlink替代gitlink记录。
将[repro](repro/)脚本放回私有根`artifacts/`，先build，再compile，再执行：

```bash
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 \
  --run 'bash <private>/artifacts/run_runtime_dma_entry_20261012.sh'
```

构建脚本显式使用PYPTO_BUILD_JOBS；MIX因使用当前进程编译hook，以pytest `-n 0`在同一张卡串行执行。
增量patch使用`git apply --unidiff-zero`；runtime按versions中的gitlink更新，保持私有SDK各组件版本一致。
