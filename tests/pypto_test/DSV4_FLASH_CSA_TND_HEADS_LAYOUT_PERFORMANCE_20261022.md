# TND精度版 Heads 数据通路：先性能、后精度筛选（2026-10-22）

## 结论

本轮完成两个独立布局候选、三个短任务。**8K/B16有约2%的稳定完整图收益，性能筛选后的Native逐位对拍通过；128K/B24只有约0.6%的完整图收益，低于预设1%门槛，未做该档精度验收。**

- A Direct：attention最终结果直接写Native heads，删除分组缓冲及独立复制。
- B Packed：保留attention分组输出，WO-A直接读取它，删除Native heads复制；11处寻址更新，真实T显式传入。
- 冻结对照、原环境和默认路由保留。两者是互斥单因素，不能把收益相加。
- 只生成和测试独立诊断文件，尚未把候选合入正式算子。没有修改SDK、runtime或编译生成的C++。
- 本轮是第4层真实权重、合成hidden/history的单层完整HC-pre→HC-post测试，**不是新的整模型192-token验收**。

## 完整图性能与精度

单位μs，NPU event中位数；每个任务12个交错窗口，每窗口100次raw FULL replay。
“降低”定义为`1 - 候选耗时 / 冻结耗时`。不同任务分别报告，不跨任务拼接收益。

| 配置/任务 | 冻结版 | A Direct | A耗时降低 | B Packed | B耗时降低 | 性能筛选后的Native精度 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8K/B16，第1轮 | 1308.24 | 1281.79 | 2.02% | 1277.46 | 2.35% | 两者A/B八边界逐位通过，ABA通过 |
| 8K/B16，第2轮 | 1312.84 | 1285.64 | 2.07% | 1283.42 | 2.24% | 两者A/B八边界逐位通过，ABA通过 |
| 128K/B24 | 1990.20 | 1977.89 | 0.62% | 1978.23 | 0.60% | 两者未达性能门槛，跳过精度验收 |

每个候选在每项任务中均12/12窗口快于冻结版。8K两次独立任务各自复现，合计每候选24/24窗口领先。
A、B之间完整图差距仅约0.17%～0.34%；当前证据不足以认定Packed明显优于Direct。

128K不是完全零收益，但没有达到任务前固定的筛选条件，按用户要求不继续投入该档精度验收或后续整模型任务。
`result.json`中的`pass=true`在该档表示性能筛选流程和执行有效性检查完成，**不表示128K候选精度通过**；应同时查看`accuracy_executed=false`和`accuracy_skipped_for`。

## 核内L1

单位μs，独立5窗口，每窗口1次raw FULL replay，按72核最早incore开始到最晚结束重算，包含内部等待及阶段间隙。

| 配置/任务 | 冻结版 | A Direct | A耗时降低 | B Packed | B耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K/B16，第1轮 | 1207.26 | 1180.92 | 2.18% | 1181.96 | 2.10% |
| 8K/B16，第2轮 | 1212.12 | 1182.76 | 2.42% | 1176.62 | 2.93% |
| 128K/B24 | 1924.46 | 1889.08 | 1.84% | 1875.52 | 2.54% |

L1与event是不同采样口径，不能相减各自中位数来推算AICPU耗时，也不能仅凭L1改善判定完整图通过筛选。
本轮没有重新测试nalinaly，不把此前截图数字混入同环境对拍。

## 身份与实现变化

正式分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，冻结源码提交`ed2b2e5c3c88a6d8682a6ad2ab8262686bcc2dc3`，QA168修正`ad06a423bc889ca1730a726303845625df29d91d`。
直接复用上轮canonical源码、原SDK和缓存：3399文件manifest `48ee0351…`、77运行源清单`e0b25682…`、QA `ea30b2de…`、原编译收据`aee68650…`。
完整哈希与模块路径在每包的`binding.json`、`SOURCE.json`、compile receipts和实际preflight中。

两候选的数学AST经独立逆向归一化与冻结版一致；没有改MMA形状、Native packet起点和累加次序、WO-A BF16边界、8192元素整行量化或WO-B。
A只改最终BF16 store接口/坐标；B只改heads输入布局/寻址与行数来源。
独立地址检查覆盖139264个packed有效病例及T1～256的发布映射，但这不是对应全部T的NPU精度证明。

两者均通过不初始化NPU的完整根、内核、编排编译；实际编排确认根父缓冲由merge输出、后续WO-A读取，merge保留QK及RoPE依赖，独立复制任务已删除。
外部根仍为64Tensor，完整替换HC-pre→HC-post，计算全部使用DSL。

## 测试方案

### 输入与环境

- 模型：正式DeepSeek-V4-Flash-0731-w8a8第4层权重，仅单层，不加载整模型生成token。
- TP1，page32，NZ2，batch16或24，全S6，真实T96或144；history8192或131072，seed1024。
- 使用Native builder创建metadata、compact RoPE和独立cache/state；各arm输入/初态相同，循环逆序页表。
- Indexer历史scale为`0.00390625+((row*37)%251)/16384`。
- 原隔离Python3.12、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
- Native确定性Torch/NPU level1/HCCL开启，13个官方静态包实际安装/reselect，DSA多流，SK0。
- 真正raw FULL图，禁止PTO回退；检查实际PTO dispatch和64Tensor schema。

本矩阵专门为全S6；代码仍使用真实T和Native请求边界，但不新增短query/混合1～6行或整模型token资格。
Native先捕获仅为静态资格和后续精度reference；筛选前不比较Native/PTO数值，也不使用Native作为本轮性能对照。

### 先性能

1. 四arm依次准备Native、冻结、Direct、Packed，各自独立cache，保留原始权重存储。
2. 三个PTO arm warm3；每个交错窗口外restore cache/readonly并同步。
3. 12组event，每组100次完整raw replay，六种arm顺序各两次。保留全部36样本，不删离群值。
4. 另采5组交错L1，共15个原始trace；dep-gen关闭，没有替换生成C++等待helper。
5. warmup和每窗口结束、下次restore前检查finite、33项guard/metadata、readonly权重包；每任务54检查点。
6. 任务前固定的筛选门槛：event中位数至少快1%、至少10/12成对窗口领先，且L1中位数也更快。

这是筛选门槛，并不是把小于1%的实测变化改写为零；8K另用第二个独立任务验证稳定性。

### 有性能收益后再测精度

性能结果先写入`result.json`并打印`PERFORMANCE_SCREEN_COMPLETE`，之后才开始Native数值比较。
只对达到门槛的候选及冻结对照做：

- A输入下，输出、TopK、六类cache/state共八边界逐位比较。
- hidden扰动`+0.125`后的B输入做相同八边界比较。
- 恢复A重放并与该arm先前A逐位比较；检查guard与readonly，首差立即停止。

两轮8K每轮共48项Native边界比较通过，Native及三个PTO arm的ABA通过。
128K两个候选未进入该步骤，不能引用8K结果推断它们128K逐位或逐token一致。

## 任务与复现

| 配置 | 任务 | 入口计时区间 | 结果 |
| --- | --- | ---: | --- |
| 8K/B16，第1轮 | `task_20261022_103841_168933531766` | 210秒 | completed exit0，已释放 |
| 8K/B16，第2轮 | `task_20261022_104327_182736017241` | 148秒 | completed exit0，已释放 |
| 128K/B24 | `task_20261022_104719_20147496213` | 206秒 | completed exit0，已释放 |

每个任务单卡，外限780秒、入口内限750秒；没有长时间独占卡、后台watch或定时自动化。
原始提交命令、完整启动脚本、最终日志、编译和实际环境记录均随证据归档。

同一已准备环境中，可用证据内助手复现；每次使用新case名防止覆盖：

```bash
python task.py submit heads-8k-b16-v3 16 8192
python task.py observe heads-8k-b16-v3
python collect.py heads-8k-b16-v3
```

`task.py`仅支持submit/observe；实际采集使用`collect.py`，只在任务completed后运行。
128K使用`batch=24 history=131072`；不自动重试未知提交结果。
首次准备环境时，按`env.sh`、`preflight.py`核对精确导入身份，先用`compile.sh direct/packed`完成无设备编译；
保存源码/编译收据，再运行`prepare_probe.py`生成helper manifest。不要覆盖已验收环境或直接占卡编译整套SDK。

完整文本证据和CPU复核说明见[证据目录](evidence_tnd_precision/heads_layout_perf_20261022/README.md)。原始trace无权重、cache payload、失败Tensor或构建二进制。

三任务共108个event样本、45个原始L1窗口、162个执行检查点已独立复核。
入口区间不包含任务开始前的环境准备；全部任务外限780秒，所有调度终态均exit0。

## 取舍

8K的两种小改动值得保留为候选，其中Direct代码改动更少。当前没有证据证明Packed的额外输入布局修改带来明确更多收益。
128K完整图收益不足，因此本轮不作为跨配置通用优化合入，也不提交整模型任务。
下一步更值得测试的是保持Native数学顺序的attention最终heads融合或核内状态复用；继续先性能筛选，再为赢家投入精度验证。
