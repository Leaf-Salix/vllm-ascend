# 长档Score分片均衡候选

固定基线8e176285，独立源码`.cache/csa-score-balanced-8e176285`；只改性能版Indexer，
未合入生产，也不混入正在执行的8e176285三档模型验收。主要覆盖128K/B8与B16形成的16个query组。

## 来自AscendC的依据与PTO的取舍

最新ops-transformer b5b33e14的QLI V2按2048候选粒度估算并分配核间工作，
metadata的AssignByBlock考虑剩余代价，Vector保留累计Top-K。
PTO当前按固定8192候选leaf轮转分给24个worker。128K decode约32769个可见候选时，
16个query组各有8/8/8/8/1个N1024步，最忙worker执行25步，最少17步；
这部分差距来自核内循环工作量，不能全部称为调度延迟。

候选保留现有N1024、Key预取、FP16/Cube算术及分页视图，把16组的leaf数量向3的倍数取整，
再均分N1024步。32769候选得到6/6/6/5/5/5步的六个leaf，24个worker各22步，
最忙核工作步数减少12%；32768候选则24→22步。这是静态工作账本，尚不是实测加速。

没有直接照搬Native每2048候选发布一对根：那会扩大pair arena并增加归并。
本方案继续使用现有8192列score arena、每query最多32个leaf及64个half-root槽；
近最大支持长度若取整超过32个leaf，就回到原计划。非16组及短档Score分片保持。
长档merge按同一全批次计划计算每个query可见的根前缀，不能按旧8192公式读取未发布槽。

代价是128K的leaf由5增至6，query/系数加载、AIV排序及最终归并可能增加：
10个half-root的三轮四路归并变为12个根的四轮。需要同时看AIC/AIV均值、最大核、Score包络和完整CSA，
不能只依据更均匀或最忙步数下降保留。
相同score的合并分组发生变化，若Top-K排列不同必须定位到明确tie规则，不能把状态差异直接忽略。

pypto-lib仍作其连续cache/query组织参考；本候选保留Native分页零复制，不新增入口拆分或写回。
与上游的分组及分片差异是为了当前A3上16组/24核的工作平衡，是否值得保留由单卡及最终模型决定。

## 验证范围

[补丁](candidate.patch)基于8e176285；[CPU分片账本](partition_evidence.json)只核对连续覆盖、
可见前缀边界、最大支持长度回退和pair容量，不能代替设备正确性。
Ruff通过；[全CSA CPU编译入口](compile.py)由既有监视器确认当前模型结束后执行。
首次在ConvertToSSA失败：两个leaf-plan标量只在constexpr分支内定义；
已在算子侧给出分支前默认值，重新编译，不改工具链。
[失败摘要](compile_scope_failure.txt)保留，完整首轮IR日志在本地compile_scope_failed.log。
随后遇到整数边界`[1,0]`错误，仅将leaf步数限制为正数仍未解决；
将可见根数改为两个前缀计数之和，消除零长度前缀的条件分支后，完整lowering/PTOAS/CCE/链接/load通过。
两次边界失败分别保留在compile_bound_failed.log、compile_bound_clamp_failed.log；
成功[日志](compile.log)及[CPU/设备不重叠记录](compile_guard.jsonl)均保留。

CPU通过后先用[定向单卡入口](run_layer.sh)对比128K/B16和8K/B16，各20次无profiler图计时，
四个独立DFX窗口、八类状态和A→B→A重放；保留P95及全部异常样本。
已提交auto单卡`task_20260928_203745_30431952702`，最长2400秒；尚无设备结论。
如果有可保留收益，再补同策略B8及必要tie/边界缺口，不直接重跑七档或扩展EP16。
按用户最新优先级，本阶段主看128K incore与CSA调度、8K控制及七三指标，整网性能验收后置。
