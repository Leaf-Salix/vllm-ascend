# 同算术 CSA 对照与已就绪门铃候选

2026-10-12，过程073。局部 `wait_pre_task_end()` 继续手工标识，chip swimlane、PMU、参数 dump
仍使用串行回退；两项均是用户确认的范围，不列为缺失能力。

**结论：补齐同算术普通路径；已就绪门铃快路径没有形成净收益，不保留。**
完整 early-start 的稳定净收益尚未成立，功能覆盖与性能结论分别记录。
公共 CSA 默认未修改；没有新 Native、七档、16卡或 token 对照。

## 改动与对照依据

过程071的普通路径仍有 Score 重复扫描，不能与已消除扫描的 early-start 直接归因比较。
本轮从保留的 `deepseek_v4_flash_csa_score_scalar_plan` 冻结副本生成普通路径
`deepseek_v4_flash_csa_score_plain_control`，仅删除10个文件中的57处 Wait、55处 Set。
两侧使用相同最大长度 Scalar、数学表达、tiling、metadata、策略及 `allow_early_resolve`。
具体删除见 [源码差异](fair_before/compiler/plain_vs_early_operator.diff) 和
[语句计数](fair_before/compiler/removed_boundary_statements.json)；73个生成成员的前缀检查通过。

候选仅修改 Simpler AICore 执行器：没有本核未完成前驱的 gated 任务，如果完整 doorbell
已经等于自己的 task ID，则走原门铃/EXIT 检查及 ACK/入口，省去延迟回调。
门铃未开、仍有本核流水和 MIX 成员仍保留原安全前缀路径。
候选没有改变队列、核选择、dispatch、ACK后行为或用户 early-resolve 表达。

代价是符合条件的 gated 入口新增一次门铃读取。已经命中后，原普通循环仍重新检查门铃；
这部分确有重复读取，但本轮没有分离测量它的耗时，不能把全部差值归因于此。
候选源码见 [Simpler patch](simpler-candidate.patch)，未进行后续参数搜索。

## 128K/B16 性能

单卡0，第2层完整 HC_pre+norm+CSA+HC_post；CANN9.2、TMR、NZ2、atomic0、deterministic0，
关闭上述诊断。每轮预热5次，两路各30次轮换单次 NPUGraph replay，外部 NPU Event 计时，reset 在区间外。
两轮分别内部对照；不能将跨轮 mean 差相减作为候选的精确因果收益。

| 轮次/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 修改前：同算术普通 | 974.60 | 993.49 | 1016.18 |
| 修改前：原 early-start | 962.44 | 993.69 | 1011.76 |
| 门铃候选：同算术普通 | 969.22 | 990.80 | 1015.76 |
| 门铃候选：early-start | 977.32 | 996.55 | 1017.16 |

修改前 early-start 相对普通 mean 多0.20μs，配对中位差−1.48μs，16/30次更快；max 少4.42μs。
候选相对同轮普通 mean 多5.75μs（+0.58%），配对中位差+5.56μs，10/30次更快；max 多1.40μs。
候选没有净收益证据，撤下，不将一次测量泛化为此策略必然更慢。
修改前结果也只能说本轮均值基本持平，不能声称已获得稳定加速。

全部120条样本分别见 [修改前结果](fair_before/results.json)、[候选结果](candidate/results.json)；
配对统计及功能摘要见 [summary.json](summary.json)。

## 功能与真实分支覆盖

两轮无插桩计时路径的9项输出/状态、改变 hidden 输入后的图重放均逐 bit 一致，两侧各31项保护区通过。
首次设备任务中两项 MIX 通过；四项跨核用例在旧 trace 断言失败：低八位类型7之上新增了通知标志，
合法值263被旧 `== 7` 误拒绝。修正为分别校验类型与允许标志位，数值、完整 fanin、本核尾部、
sync_start 和保护区断言未放宽。只复跑这四项，全部通过；首次失败证据保留。

独立插桩使用进程内 trace arena，只作覆盖诊断，不混入计时路径。两次输入各973个物理成员：

| 输入 | gated实例 | 已就绪普通入口 | 前缀回调总数 | 同核交接 | 通知提交 | 回调后交接 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 562 | 46 | 701 | 270 | 707 | 21 |
| 1 | 582 | 23 | 772 | 313 | 671 | 28 |

前缀回调总数含正常 ready 派发的同核交接，不能直接与 gated 实例相减。
快路径仅覆盖两轮 gated 实例的8.19%/3.95%，其余 gated 实例仍走回调。
两轮9项状态及31项保护区均通过，完整尾部/消费者等待时序断言通过。
trace 没有 ACK 前的门铃时间戳，不能据此判断门铃开启时刻；门铃与 EXIT 安全性来自执行器检查。
通知尾部含新增 DMA，也不能把全部尾部等待当作原计算重叠。
[逐成员计数、handoff和原始记录](candidate/coverage/results.json)保留全部数据。

## 版本、任务与恢复

- 修改前：PyPTO `cbaf0e88f`，Simpler `6174ec5db`，PTOAS `9ddc2e25`，vLLM驱动 `c849eaee`。
- 候选：Simpler `26550cfd4`，PyPTO ABI `0935c1745`；测试断言修正 `a3d846593`，vLLM诊断 `d54ef7d4`。
- 撤下后：Simpler `9fddae626`，PyPTO `7a084029d`；Simpler内容与 `6174ec5db` 完全一致。
  已从私有 worktree 重建绑定、TMR runtime 和 Torch 适配器，三者 revision 对齐；不重复测未变代码。
- PTOAS与PTO-ISA本轮未改；完整版本和任务映射见 [retained_versions.json](retained_versions.json)。

| 任务 | 用途 | 结果 |
| --- | --- | --- |
| `task_20261012_023823_159410115924` | 同算术修改前对照 | exit0，设备释放 |
| `task_20261012_024557_188858716474` | 候选小用例 | 两项MIX通过，四项旧trace断言失败；exit1，设备释放 |
| `task_20261012_025337_21671581309` | 四项修正后重试、CSA计时、独立覆盖 | exit0，设备释放 |

源码驱动和增量 patch 在 `source/` 与本目录；编译、首次失败和重试日志均已保留。
`repro/` 脚本依赖此前的私有环境、冻结CSA包和固定权重，需放回私有根目录 `artifacts/` 使用，
不是新克隆仓库上的独立测试包。CPU编译不占卡，所有NPU执行经 `task-submit --device auto` 分配。
候选需检出对应版本并重建；当前恢复后的SDK不可混用旧revision的计时产物。
零上下文patch需 `git apply --unidiff-zero`，不覆盖公共算子。

## 保留项与后续边界

保留同算术对照驱动、已修正的trace断言和可解释已就绪入口的诊断分析。
保留已有低开销通知协议及AIC/AIV/MIX、idle、sync_start的安全前缀能力。
一般只读GM前缀尚缺写者/别名合同；事件耗尽的阻塞回调期间尚不能继续监测后到任务。
这两项仍是实现范围限制，不能改称用户接受的限制，也不能因为功能子集通过而宣称目标全部完成。
