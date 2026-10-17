# 主Query DQ穿插Score消费：保留设计，本轮不实现

**结论：源级依赖允许构造无环的4-head分段，但现有收益证据不足，本轮不实现、不编译、不上卡。** 同compact优先顺序的旧长B16观测显示，DQ大部分已与主压缩投影重叠；移走它可能主要换成更早等待MAIN_PROJECT。新增重复准备、跨Score存活的UB与三文件调度改动，相比当前证据不值得继续推进。下面只保存一个具体设计供未来有新证据时参考，不形成候选源码。

只读起点：正式668的 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/` 33源。没有叠加正在测的owner或Attention geometry候选。所有行号指该正式源。

## 排重和时序证据

未在MASTER、ATTEMPTS和现有专项报告中找到“保持主压缩顺序，把主DQ分成4-head段插在每步Score ACK后”的设备试验。相关旧负例并非同一方案：

- `tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md:1896` 与 `FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md`：把主Q后处理整个阶段放到Indexer后，固定成本模型773.41→788.94μs，Indexer最后完成不提前，Attention后移；没有生成设备候选。`deferred_main_q_schedule_analysis.json` 的799.38对773.43也只是另一旧成本模型。
- 主PAR `qb_dq_handoff_readonly/README.md` 否决逐QB列通知：DQ入口时全部QB早已结束，新增信号不能提前其入口。本设计保留全QB等待，区别是尝试在Score内部执行独立DQ工作，不是重做逐列通知。
- `compact_priority_evidence/README.md` 的已采用版本仍为AIV `COEFF→compact quant/scale→QUERY→MAIN→Score`，没有实现本分段。减少等待通知的旧消融也不能当这项设计的实测。

现成d253长B16三轮 `parallel_20261018/attention_tick_h131072_b16_v1/clock_round_{0,1,2}_spmd_reference/phase_clock.json` 使用compact优先顺序；T96的16-head DQ分组与正式668相同。读取结果保存于 [evidence.json](./evidence.json)。

| 同轮通知端点/μs | 第0轮 | 第1轮 | 第2轮 |
| --- | ---: | ---: | ---: |
| MAIN_PROJECT全部24 AIC最晚结束 | 296.68 | 292.32 | 291.12 |
| QUERY全部48 AIV结束中位 | 299.14 | 294.36 | 295.06 |
| 中位QUERY端点减去全MAIN端点 | 2.46 | 2.04 | 3.94 |
| QUERY最晚端点减去全MAIN端点 | 5.34 | 4.78 | 6.20 |

原DQ区间逐AIV中位为22.36/23.21/24.30μs，其中全QB检查约3.37/3.08/3.24μs。主压缩pool在 `vector_schedule.py:779` 必须先取得全部MAIN_PROJECT owner；故删去前置DQ并不意味着pool/Score可以提前上述整个DQ区间。约2–4μs是**旧同序版本的端点观测约束**，含通知/缓存维护/插桩，既不是正式668的实测收益，也不是任何重排的精确可省上界。资源竞争和异步推进变化还可能改变这些端点。

旧593的AIV Score READY累计等待约107μs没有逐步空隙分布，也不是当前正式668；不能把这107μs当可容纳DQ的免费预算。现有证据足以降低本方向优先级，但不证明其必然无收益。

## 唯一保留设计：原owner内固定4-head小段

原 `query_projection_stages.py:174–200` 以实际T选择head_tile∈{4,8,16,32,64}，原任务 `work=(row_tile*(64/head_tile)+head_group)` 归 `work%48`。保持此规则及每核原任务顺序，**不能重新把所有4-head小段编号后再取模**。

每AIV维护两个Scalar游标：`dq_work=worker+k*48`、`head_offset=0,4,...,head_tile-4`。该小段处理原 `tg=(dq_work//(64/head_tile))*8`、`hg=(dq_work%(64/head_tile))*head_tile+head_offset`。完成4头后仅推进offset，原任务内全部头完成才推进work。Score中的 `worker=2*buf_worker+buf_score_lane`，不是group/query编号。

建议接口只是设计签名，不创建文件：`q_dequant_rope_chunk_stage(原7个GM Tensor, dq_work, head_offset, head_tile) -> q_flat`。helper按原公式计算tg/hg，完整8行保留原逐head `pl.pipeline(...,stage=2)`，一次4头；尾行保持原4-head `recip(sqrt)`路径及valid写回，不改成完整块的rsqrt。每head的INT32→FP32、先激活scale再权重scale、512列归约、RoPE、rint和两片写回顺序不变。没有未完成head跨切点，也不携带DQ Tile穿过Score。

`indexer_score_vector_lane_stage` 需要接收这7个已有GM引用、原worker/游标；新增的是callee参数，不是根Tensor或新GM。为了保持单个Score叶的UB root/segment，不能每步返回到外层Python或把Tile当接口实参；小段应在同一个inline Score helper内部调用。

长balanced仅在 `indexer_score_stages.py:469–590` 的本步READY→load→ACK→scale/assemble→必要sort/merge**全部结束后**插最多一个小段，再做下一步scale准备/READY。不放在READY前首次等待之前，也不在load和ACK中间。游标跨叶不重置；Score全部结束后补齐未做DQ（含没有Score工作的AIV），随后外层调用原 `publish_phase(PHASE_QUERY)` 一次，再发布 `PHASE_SCORE_VECTOR`。QUERY发布语句必须从原DQ后位置移至此处，不能声称源码位置不变；其所有者、代际、发布维护及后继全QUERY获取保持不变。

## 依赖、共享state和空任务

保留 `vector_schedule.py:761` 全24 QB获取，QR/ROPE获取 `567–590` 也已经完成。DQ只读Q_B输出、QR scale和RoPE输入，写独立q_flat；它不读写Indexer Key/scale、root、主压缩state或Score环槽。主压缩的MAIN_PROJECT获取、pool发布、全真实pool owner获取、commit/output顺序（775–824）逐句保留，不能把DQ穿插进共享state读写之间来绕过pool集合。

AIC在进入Score前已完成本核QB和MAIN_PROJECT（`cube_schedule.py:313–353`）；Score只依赖Indexer Query/coeff/scale，不等待PHASE_QUERY。AIV的所有DQ小段前驱在进入Score之前已经获取，不在持有Score槽时等待新的producer。现有两槽ACK和每leaf drain不变；有限DQ段可能延迟下一ACK，但不会形成环。严禁加入 `PHASE_SCORE_CUBE` 整段完成等待。

DQ完成并发布QUERY后才发布SCORE_VECTOR，因此后继全Score→TopK→plan获取链仍包含全部DQ；AIC `cube_schedule.py:381–383` 的plan及全QUERY等待保留。原q_flat与q_heads仅是声明过的GM别名，无提前Attention消费。dummy、S1/S5、mixed Q仍要按真实T完成原DQ写入；不能用Indexer live query数省掉主Query。Score无任务的核直接在尾部补齐DQ并发布，不伪造ACK。

## 容量、重复工作与长短界限

既有DQ生成物 `parallel_20261018/query_head_group_evidence/generated_audit.json` 的DQ Vec地址末端为131072B；这是地址包络，不是运行时live峰值。balanced G6的持久root为3×1024×4=12KiB，segment为6×1024×4=24KiB。把DQ原包络与这36KiB隔离，概念预算为164KiB，距184KiB预算20KiB；不新增L1/Left/Right/Acc或GM容量。

这尚不是可放下的编译证明：原root在UB0、segment在65536，与原DQ包络冲突，完整planner必须重新安排；如果本步scores/scale/sort临时量在插入点仍存活，或者跨循环phi延长生命，20KiB余量可能不足。只能在已处理完本步分数之后插入并让本步临时量死亡；不能直接沿用旧地址。全MIX的事件编号、PIPE_ALL、DMA供数与UB规划仍可能改变。

每次完整8行准备读取逻辑QR scale32B、cos/sin/swap各2048B，合6176B，还包含arange、行偏移和gather索引准备。4-head切分的额外成本为 `(head_tile/4-1)×6176B/原任务` 及对应准备指令；不把这些开销设为零。

| 例子，仅说明公式 | 原每AIV head/任务 | 分段 | 全48 AIV额外prepare读取 |
| --- | --- | ---: | ---: |
| T96、head_tile16 | 16头/1任务 | 4段/核 | 889344B（868.5KiB） |
| T192、head_tile32 | 32头/1任务 | 8段/核 | 2075136B（2026.5KiB） |
| head_tile4（含尾行） | 原任务均4头 | 每任务1段 | 0；仍有调度和调用开销 |

逐head投影读取、权重scale读取和q_flat输出字节不变；上表是逻辑载荷，不等于HBM额外流量。原整波扩组的意义正是减少这些准备，分段会局部放回这部分工作。

若未来重新考虑，最小范围应先仅支持现有 `BALANCED && QUERY_GROUP==6 && SCORE_TILE==1024` 长历史路径，依据是明确的持续Score流和root/segment预算，绝不是B16白名单。128K/S6 B16有22次Score消费/核、4个DQ段/核；这说明有插入位置，不说明每个位置存在足够空闲时间。低B长历史5–11步也无必然空隙；mixed长度可不均衡，必须尾部补齐。短G2/768原本半叶GM路径不同，6/12步分别面对4/8个DQ段，且末尾半叶排序时机不同；不凭长档设计直接开启短档。长G2/S1同样走非balanced，应保留原路径。长G6的S5及mixed尾行能用原尾数学表达，但需要独立边界验证。

当前 `system_ops.py:119–139` 只有阻塞 `sync_wait`，没有本设计可用的READY无损peek。固定插段并不依赖非阻塞API，所以不构成语义障碍；它可能在两槽已有积压时抢延下一次ACK，或与AIC搬运争抢带宽。两lane按各自原DQ工作推进，较慢者仍能拖住共同ACK，不能按平均DQ量假定平衡。

## 若将来有新证据，最小验证范围

只需先准备正式668隔离reference/candidate，constexpr默认关闭，仅long G6/balanced开启；动态T和Q/L/position/页表保留。一次完整long符号CPU检查该路径、原数学AST/owner、4-head stage2、尾行、真实UB/事件和非目标关闭分支，不扫参数、不预先扩大短策略。CPU是需要另行授权的后续事项，本轮没有创建源副本或执行命令；即使通过也不能预记性能。

现阶段不推进：旧同序端点已经提示前置DQ大量处于MAIN_PROJECT尚未完成的时间内，而可穿插的逐READY窗口没有现成测量；收益依据弱于新增准备/活跃UB/调度复杂度风险。保留本设计和原始观测路径后收口，不再扩模型或申请设备。
