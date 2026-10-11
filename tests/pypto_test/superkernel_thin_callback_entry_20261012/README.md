# 过程066：通知入口栈帧检查与小入口候选

2026-10-12。私有A3/TMR、单卡128K/B16。本轮候选功能通过，但没有净性能收益，不接入。
PyPTO实验驱动`40a753dd4`，私有vLLM驱动`3a053b43`；生产生成器仍为`21a3fbcfe`，
Simpler `5085a6ce0`、PTOAS `0c21f984`不变。[完整版本与增量](versions.json)。

## 目的及实现边界

过程065记录了1329次通知、42次完整回调后交接，因此继续检查通知调用成本。
现行路径从运行时回到当前task的`kernel_entry(args | 2)`，再调用`pypto_poll_local_tail`。
先检查是否能仅用强制尾调用减少调用成本，再试验将通知与普通计算体分开组织。

`thin`仅在实验编译hook中改写生成的C++：普通任务体放进内部noinline函数，
保留其完整参数解码、交接位、Scalar前缀、Wait、模式初始化和算术；`kernel_entry`只分流两种入口。
通知仍使用原64B DMA、完整流水汇合、事件与缓存可见性规则。每个task仍独立编译，
没有预编译前后两个task的组合，没有修改AICPU队列/派发/ACK/核分配或用户early-resolve。
算子作者仍仅使用局部Wait/Set，chip swimlane、PMU及参数dump回退保持。

相比原版，通知入口可能少用一层大的栈帧，但普通计算体增加一次函数调用。
不能只看通知入口变小便认定整体受益，必须测完整CSA。

## 编译证据

使用正式构建的`ensure_pto_isa_root()`解析pin、相同CCEC参数和SDK include，
以`-fstack-usage`记录每函数静态栈大小，并在链接后用SDK ELF解析器比较代码。
三个代表kernel的`musttail`与原版链接后`.text`逐字节相同，因此不上卡重复测试这一候选。

| kernel | 原入口栈/字节 | musttail入口栈/字节 | thin入口栈/字节 | thin计算体栈/字节 | 通知helper栈/字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| csa_cache_writeback | 88 | 88 | 16 | 88 | 8 |
| indexer_score_topk_native_pair_aiv | 320 | 320 | 16 | 320 | 8 |
| _proj_b_mm_nz_kernel__2 | 216 | 216 | 16 | 216 | 8 |

静态栈大小不是特定分支实际耗时，也未证明旧入口所有栈操作都会在通知分支执行。
thin三个代码体分别增加36、36、64字节，没有对通知区间单独计时或将差值按栈大小比例归因。
[编译命令及栈记录](cpu/results.json)、[链接后代码比较](cpu/linked_comparison.json)、
[入口改写差异](cpu/entry_changes.patch)。未复制完整CSA计算体或设备二进制。

探索阶段的失败保留在[failed_attempts.json](cpu/failed_attempts.json)：CCEC不支持设备侧`-S`；
手工调用最初遗漏编译工作目录使相对include失效；首次成功比较使用的非pin路径被弃用，
随后全部用正式pin重编。系统readelf不支持部分设备DWARF，最终证据用SDK解析器取得。
这些失败均发生在CPU构建检查阶段，没有错误构建进入设备测试。

## 单卡功能与性能

任务`task_20261012_001136_18204832355`，卡0，退出0并释放设备。
先运行AIC/AIV小用例，loops=4、额外Scalar循环=0，两组输入及图重放；
真实fanin读取、输出、保护区和完整完成顺序均通过。后到交接按原首次探测后的trace位统计：

| 核类型 | 输入0后到交接 | 输入1后到交接 | 图重放后到交接 |
| --- | ---: | ---: | ---: |
| AIV | 1 | 0 | 1 |
| AIC | 1 | 15 | 13 |

计数关联同核、同generation实际消费者，不将含通知DMA的尾等待当作原计算重叠时长。
[小用例完整结果](small/results.json)。

完整CSA使用第二层真实权重、128K合成历史、B16×verify6；区间为HC_pre+norm+CSA+HC_post。
CANN9.2、TMR、NZ2、atomic关闭、deterministic level0，无诊断；每路预热5次，
随后轮换各30次单次NPUGraph重放。使用NPU event计时，状态reset在计时区间外，单位μs：

| 路径 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通PTO baseline | 971.78 | 991.25 | 1012.04 |
| 原完整通知 mixed | 966.10 | 988.23 | 1006.40 |
| 小入口 thin | 964.02 | 989.82 | 1006.82 |

thin相对mixed mean多1.59μs（约0.16%），max多0.42μs；16/30配对更快，中位配对差−3.01μs。
不同指标没有一致收益，不采用，也不追加重复测试以寻找有利结果。相对普通版mean少1.42μs，
但15/30配对更快、中位差+0.63μs，不能据此宣称完整CSA已有稳定加速。
正常路径新增调用可能抵消通知入口的节省，但本轮没有独立测出两者各自开销，不能将其定为唯一成因。

9项输出/缓存/状态、换输入重放均逐bit一致，每路31项保护区通过。这是三种PTO实现间的检查，
没有新增Native、七档或16卡验收。[90条样本](timing/samples.csv)、
[完整结果](timing/results.json)、[配对差值](timing/paired_comparison.json)。

## 保留与复现

保留当前完整通知实现，候选仅存为私有实验驱动；公共默认不变，稳定CSA净收益仍待解决。
下一步不重复musttail、helper直接内联或整任务体外提这几个已验证方向；
继续检查能够减少通知公共成本、同时不增加普通任务调用开销的实现。

在已有冻结CSA包和过程063编译产物的私有环境中，按[versions.json](versions.json)确认工具链版本。
本归档保存本轮增量，完整历史私有包需沿用前序阶段产物。
将[repro](repro/)脚本复制回私有根的`artifacts/`，
先执行`compile_thin_entry_20261012.sh`完成CPU预编译，再执行：

```bash
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 \
  --run 'bash <private>/artifacts/run_thin_entry_20261012.sh'
```

CPU检查脚本为`inspect_tail_calls_20261012.py`；从私有PyPTO仓source `../env.sh`后执行。
基线和原通知复用过程063的compiled目录，候选独立编译；原产物路径在结果及reused_artifacts记录中。
增量patch按`git apply --unidiff-zero`应用；不要把编译期插桩产物替代正常计时产物。
