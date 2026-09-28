# 554b3bca阶段七档验收

冻结源码`.cache/csa-key-l1-seven-554b3bca`，性能版纳入独立Key L1池、Top-K UB和修正后的QR输入驻留。
不包含主工作树其他会话未提交的WO_A/Runner改动。当前目标优先128K，8K代价按长短变化率7:3评价；
七档按128K三档、8K四档各自等权后再7:3，P95、token、DSpark独立列出。

先单卡、再真实权重EP16；同权重、mode2、atomic0、det0，EPLB关闭。
单卡用第二个CSA层（layer4）真实权重/合成历史，反向物理页与逐物理行Indexer scale。
模型只计预热8步后连续10步纯decode forward，另采独立rank0三步PyTorch profile。
模型阶段启用已准备的metadata builder子区间诊断；它是嵌套主机观测，不扣除设备forward中的等待。

## 单卡覆盖

长短B16已对同一算子实现完成状态/图重放、20次计时和四窗口，直接复用：
[128K/B16](../csa_score_key_l1_20260928/h131072_b16/evidence.json)、
[8K/B16](../csa_score_key_l1_20260928/h8192_b16/evidence.json)。
该候选`decode_indexer.py`原样保留到554b3bca；其余生产算子与2d2f9ca0相同。
只新增可选模型主机观察，与单卡CSA实现无关，不将不同算术或不同策略版本拼表。
原任务与缺失短档DFX补采的来源、失败记录均保留；不重新跑两档以挑选更好样本。

[其余五档入口](run_remaining_layer.sh)：128K/B4、B8，8K/B24、B32、B40。
单卡任务`task_20260928_181207_305949327631`已完成、退出0；五档计时与独立DFX齐全。
每档一次Native/PTO计时、候选自重放/保护区及A→B→A，再采两个独立单根DFX窗口。
这五档是阶段覆盖，不重复四窗口单因素实验，也不额外导出大块cache状态供无用途比较。
Native零容差浮点/量化差异仍诊断记录，不冒充逐bit一致；最终token/DSpark由整模型验收。

阶段结束保留七档全部模型PyTorch JSON及PTO泳道，汇集下载目录。
单卡七档自重放、图状态、metadata/保护区及Top-K结构通过；各档完整CSA均值/P95低于同轮Native，
按history内部等权后七三均值变化−16.306%。这不是Key L1单因素收益，也不代表整网通过。
8K/B32一个DFX窗口Score仅用18个AIC执行24份，B40一个窗口用23核，均出现同核串行；
前者启动散布79.56μs，后者30.56μs。完整计时P95/中位数最大为B32的1.036；慢样本和窗口均保留。
[七档计时与核内分项](LAYER.md)、[原始样本/全部窗口/数值诊断](layer.json)、[收集入口](collect_layer.py)。
554b3bca模型已取得128K三档，8K仍在执行；2d2f9ca0模型数字不替代本轮。

[七档模型入口](run_model.sh)要求五档单卡原任务成功退出、七档功能报告通过且分配16卡后才能执行。
已提交`task_20260928_182811_290034987`，当前运行；[任务句柄](model_task.txt)供后续查询和离线导出使用。
长档一次sweep B4/8/16，短档一次sweep B16/24/32/40，长短交换Native/PTO执行顺序。
[forward收集](collect_model.py)复用逐rank token/DSpark、位置、配置及事件门禁，七档全通过才汇总7:3；
可通过`--available`读取已齐全档位，但不足七档时不输出阶段加权收益。
[主机及builder诊断](analyze.py)保留异常rank十步标记，不根据P95好看而删除尾部。
[离线profile导出](export_profiles.sh)必须等模型任务成功结束后执行；
[报告及下载汇集](profile_report.py)生成14份真实EP16 rank0 JSON和7份单CSA DFX入口。
DFX固定选窗口0供下载，其他窗口仍全部保留，避免按性能挑图。

19:24先收集已齐全的128K三档，均通过16rank的配置/位置、token和DSpark检查，
共114688个输出token零差异、48组rank统计一致。Native/PTO forward均值：
B4为45.605/44.308ms（−2.85%），B8为55.394/55.439ms（+0.08%），
B16为72.894/69.162ms（−5.12%）。B8尚无收益；B4 P95为47.415/47.628ms，仍有尾部代价。
[正式三档样本](model/RESULTS.md)、[逐rank数据](model/forward.json)仅是当前阶段进度，
没有输出七档七三综合指标；模型profile尚未离线解析。

新增builder观察定位到B4 PTO step15/rank14的C128 attention `build_decode_metadata`：
墙钟3.703ms、线程CPU3.678ms，同rank通常墙钟0.251ms；设备相对入场迟到3.786ms，
自身forward正常、其余rank平均增加3.491ms。Native该builder也出现2.528ms，不能称PTO专有问题。
B4 PTO step16/rank3的另一处迟到4.534ms，现有标记显示
`batch_coordination_end → attention_metadata_begin`空隙4.449ms（通常0.118ms），线程CPU4.420ms。
该区间含deferred修正、DSA位置准备和query padding等调用，尚未区分具体调用；
step11/rank14还出现preprocess 2.039ms。以上不能统一归因于Score或添加sync_start直接处理。
[入场证据](ARRIVAL.md)、[全部分项](PHASES.md)、[异常rank全部十步](metadata_tail.json)。
本轮只在离线分析补齐现有标记间空隙，未修改运行中源码、同步、GC策略或删减正式样本。

18:46从独立B4候选的启动失败确认：当前守护进程连作业内`task-submit --status`也禁止。
七档模型当时仍pending，已移除入口中的嵌套状态查询；单卡原任务退出0及七档收集在提交前已核实，
作业内继续检查554b3bca七档PASS报告及16卡分配。仅修正shell启动门禁，不改变冻结模型源码或测量合同。
