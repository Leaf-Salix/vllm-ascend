# 完整 Indexer 链路合并实验（2026-10-27）

## 调整重点

用户指出，不应反复优化与 nalinaly 差异很小的子模块。本轮暂停 Hadamard M 维、
HC 单次屏障等微调，选择整个 IndexQuery→RoPE→Hadamard→量化→系数→Score→TopK
生产消费链。参考 nalinaly 的持久 MIX 思路，将我方原数学体放在同一个24 Cube、
48 Vector任务内，直接检验整链合并是否缩短完整 FULL graph。

此前累计 cut 定位参考为：入口至TopK+plan差约293 μs、attention区间约165 μs、
后缀约116 μs。这些是独立编译的累计切片，不是排他的模块耗时，不能相加或当作
本候选可节省时间。physical-gather-v3已有失败记录，未证明新SDK改变其lowering前不重测。

## 对照与不变项

- 正式精度运行基准11b380e3f保留，分支不切换；不写SDK、runtime或生成C++。
- 候选放在独立helper `indexer-pipeline-v6`；服务入口64Tensor、单PTO根、纯DSL、真实TND。
- IndexKey compressor先完整发布cache及scale，作为外部真实依赖；主Q_B保持外部。
- 原查询分组判据及2/6行真实请求分组保持；无跨请求分组或补S6。
- 合并scale后再乘acc、所有BF16/FP16边界、M64/K128 Had、Native系数、Score/FIFO及TopK排序保持。
- Source AST核对确认原Score整个12条顶层语句及其所有嵌套运算完全不变。
- 首版保守使用七次全体handoff，不据此声称已实现nalinaly的细粒度owner流水。
- 只准入single-leaf native Cube历史范围；128K仍走原路径，本轮没有128K候选资格。

## 生成检查

v1系数形状注解、v2返回TaskId控制流、v3 GM view放置、v4/v5动态shape嵌套Call
均在无设备阶段发现。v6显式标量表达shape、把GM reshape置于编排层，通过整个根编译和load。
这些失败没有占卡，也未修改SDK。

两臂receipt均pass，64Tensor，NPUinitialized=false；相同helper manifest：
`9c659403c5d90a672fb7879a0568c6f73a3c4d2a6ba2f6b6229286263b43933c`。
候选artifact：`33c0b4edb48f7a53eefb899196040fc62186bb15466567e50e10f4a871cb9f10`。

独立审查确认：每次只选择一个MIX，24/48角色各七次顶层同步；FFTS先初始化。
QR、scale及Indexer权重是实际任务输入，Key显式依赖最终`idx_kv_scale_commit`。
`idx_bf_rows`和`coefficient_scales`均共享原分配；TopK写原外部输出，消费者在Indexer scope退出后执行。
静态最大UB185376B，L1 360448B，L0A61440B、L0B65536B、L0C102400B。
Indexer当前depth T192约153.5MiB，T256保守约155.5MiB，低于本次默认256MiB。
该证明不覆盖128K小ring配置，也不代表整模型精度验收。

## 设备试验

任务 `task_20261027_173705_341013523268`：8K/B32/S6/T192，真实第4层权重、
合成hidden/history，完整HC-pre至HC-post FULL graph。outer780/inner750秒。
提交前一次SSH中断，随后用唯一命令`--find`确认未创建任务后才重提，没有重复占卡。

先12×100平衡event及9窗独立L1；实际注册READY/PID/generation/owner、完整CPU子项
序列化字节、编排、meta/config门禁通过后才计时，DFX不同时启dep-gen。
event中位至少下降0.5%、至少10/12配对胜且L1下降，才进入Native A/B八边界与ABA。
须反向捕获重复稳定后才可能采用，不把同进程窗口当独立试验重复。

## 结果：不采用

| 8K/B32/S6/T192，完整CSA | 冻结精度基准Direct | 完整Indexer MIX候选 | 候选耗时变化 |
| --- | ---: | ---: | ---: |
| 12×100 event中位，μs | 1605.443 | 1646.188 | +2.538% |
| 9窗独立L1中位，μs | 1521.02 | 1542.78 | +1.431% |
| event配对胜窗 | — | 0/12 | 不达门槛 |
| 对Native八边界/逐token精度 | 本轮未新增验收 | 未执行 | 性能失败后不扩测 |

实际注册身份、执行guard/固定packet检查通过。`pass=true`只表示实验流程完成；
`accuracy_executed=false`、winner为空，不能写成候选精度已通过。
正式精度基准和环境均未修改，没有新版本合入，没有跑128K或真实整模型。
本轮没有重测nalinaly或Native速度；不要把旧nalinaly参考值标成本轮共同实测。

## 大差距定位：主Q链的调度被改变

为避免将提交顺序当成设备顺序，另外提交了独立dependency诊断：
`task_20261027_174615_411357721747`。仅两臂各两窗，不混开event/L1计时。
两次映射必须相同；同时严格核对源快照、SDK及实际注册的全部child字节、编排、meta/config
与计时任务一致后，才关联既有18窗L1。每条记录均找到精确token映射：
每个基准窗1294条、每个候选窗1042条，没有用旧func编号猜测。

| 相对本窗入口时间，μs；各项独立中位 | 基准 | 候选 | 如何理解 |
| --- | ---: | ---: | --- |
| TopK发布任务/Indexer整链MIX退出 | 684.06 | 659.96 | 候选是含末尾同步的整链可用上界，非内部TopK最后store |
| 稀疏attention plan完成 | 732.94 | 726.16 | 提前约6.78 μs |
| 主Q_B全体block的跨度 | 80.54 | 349.98 | 跨核调度分布拉开，不是单核计算增加这么多 |
| 主Q_B最长单核执行 | 69.68 | 60.76 | 主Q_B二进制不变；最长单核反而较短 |
| 主Q_B最后block结束 | 461.64 | 710.04 | 输出尾部明显推迟 |
| 主Q后处理结束 | 550.96 | 771.44 | 在多数窗口成为较晚输入 |
| attention开始 | 745.10 | 782.78 | 推迟约37.68 μs |

上述区间存在重叠，独立中位数也不具可加性，不能把这些差值加成完整损失，
也不能把MIX内部未拆分的区间称作纯计算、搬运或轮询耗时。

逐窗核实比单看跨度更清楚：

| 候选9个L1窗的实际分布 | 窗数 |
| --- | ---: |
| 8个主Q_B block在MIX前完成，16个等MIX结束后开始 | 5 |
| 16个在MIX前完成，8个等结束后开始 | 1 |
| 24个主Q_B block均在MIX前完成 | 3 |

所有9窗主Q_B block均不与Cube MIX区间重叠；延后block紧随MIX退出才开始。
其中6/9窗主Q后处理晚于plan，成为attention的较晚前驱；其余3/9窗仍由plan更晚。
因此不能称所有窗口都是主Q瓶颈。

**本轮支持的结论：Indexer整链完成提前，但完整MIX改变了外部主Q链调度，完整性能回退。**
不能将回退归咎于主Q_B数学变慢，也不能仅凭scope在host代码中先闭合，就断言设备已完成。
本轮不是“某个几微秒小算子优化无效”，而是测到了整链融合与外部并行之间的结构性冲突。

## 下一项应改整套查询流水

下一版优先借用nalinaly的整套Cube/Vector分工：把主Q_B及主Q后处理一并迁入同一MIX，
移除原外部对应launch，保留QA/QR与最终Key commit为已就绪输入。
Cube发布IndexQuery后，分核执行原M64 Had与主Q_B；Vector推进RoPE、量化、系数和原Score
FIFO。各Cube做完自己的分支后，只等其Indexer输入就启动本核Score，不能再等全部主Q_B。
TopK在MIX内等待真实Score写者，主Q后处理在其真实前驱就绪后执行，最终在attention所需处合流。

这与此前失败的Query岛有本质区别：旧岛的Score在岛外，只能等全岛返回；下一方案把Score
留在MIX内，避免人工增加主Q_B→Score依赖。实现需要实际owner相位协议；本轮七个全局
handoff不是最终性能路径，不能通过继续删某一条屏障来替代整体编排设计。

分核比例、主Q INT32额外24～32MiB及动态尾部必须重新生成和审查；不直接照抄nalinaly的4核
分配为“最优”。仍保留我方Native计算和舍入，先性能筛选，收益稳定后再逐模块验精度、
8K/B40及128K/B24逐token复测。该下一版本轮尚未实现或提交，不声称已有性能收益。

## 复算、复现与结束状态

证据目录为`evidence_tnd_precision/indexer_pipeline_20261027/`。README提供离线复算和
硬件复现步骤，归档包含执行源码闭包、两份原始event/DFX/dep记录、CPU编排全文和两臂收据，
以及独立审查和任务锁释放证据。

两任务实际运行217秒、132秒，均exit0并释放设备。单卡性能失败后未做反向采用确认、
Native精度验收或大配置。单层真实权重+合成输入不能冒充整模型token验证。
