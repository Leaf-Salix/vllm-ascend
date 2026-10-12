# 过程118：关闭无使用者的调度分段计时，首轮小幅改善，等待同卡复核

从05220c71e独立构建。源码确认SIMPLER_DFX=1而SIMPLER_SCHED_PROFILING=0时，
即使chip swimlane关闭，调度循环仍调用CYCLE_COUNT_START/LAP，多次读钟并累加phase周期。
这些累加只供详细scheduler profiling日志及level≥3的phase记录使用。
候选在函数入口缓存是否需要分段计时，只在对应profiling或level≥3时读钟及累加。
总Sched窗口、超时检测、任务派发、ACK、门铃及early-resolve保留。
这是原SDK已有成本，不误归为early_start引入的开销；不叠加过程113的单处读钟候选。

构建及diff检查通过。使用实际scheduler_types.h的CPU检查确认：关闭时零读钟，
开启时保留相邻区间累计；只对这项开关行为检查，没有扩大运行时测试范围。
[源码补丁](simpler.patch)、[CPU检查](cpu.log)、[复现](repro/run.sh)。

单卡任务202358卡0退出0，全部精确检查通过。标记前/候选/后mean295.71/295.17/298.42μs，
候选比两侧中点少1.90μs；普通292.91/291.51/295.25μs。首轮结果见[results.md](results.md)。
同轮正常前→候选→正常后；固定5μs负载、30样本不变，50μs不测。
任务203350将两项候选在同一卡顺序复核，并用[DFX检查](repro/profile_check.py)确认开启profiling仍产生完整非零区间。
