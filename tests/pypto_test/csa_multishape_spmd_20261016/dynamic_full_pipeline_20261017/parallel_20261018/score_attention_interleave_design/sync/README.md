# Score / Attention 跨请求 wave 交错：只读同步与历史定界

2026-10-18，formal668。仅阅读现成源码、C++和历史报告；没有新增模型、编译、NPU任务、源码修改或hash。**存在不同于旧整helper切换的无环顺序，但当前原helper不能直接交错；资源与状态拆分尚未形成已验证实现。**

本文 `P` 为主仓 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd`，`PAR` 为主仓 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018`。下列行号取自当前正式源码及现成生成物，不是新编译结果。

## 1. 旧方案否决了什么

[旧wave报告](../ultra_wave_audit_20261018/README.md)及主仓 `PAR/wave_audit/README.md` 覆盖固定owner、三角色整helper排空后切换：B16均衡例每组仍为22个Score块加20个Attention块，零交接成本也没有缩短本组工作链。MASTER `DSV4_CSA_SPMD_OPTIMIZATION_LOG.md:16253`明确没有把该反例扩大到细粒度AIC/AIV忙闲互补。旧 `indexer_query_completion_model` 只重排叶、仍保留全阶段通知，也不等于本方案。

这里的区别只有一项：保留两套异步状态，在同组的显式循环里让**AIC发Score时，双AIV推进已就绪前波Attention**。不能把分核比例、全stage重排或“不同阶段名字同时出现”当成收益。formal668 Score三轮body中位232.28/239.39/233.21μs与旧d253 Attention供数诊断只是动机；两版本/作业的区间不能相减、相加成可省时间。

## 2. 一个有条件的无环顺序

先完成前波的全部必要叶根、TopK merge、plan发布及获取。对物理组g，用各自本地轮r执行：

| 角色 | 固定程序顺序 | 当前轮等待的数据来自哪里 |
|---|---|---|
| AIC | `ScoreStep_g(r) → AttentionTick_g(r)` | Score复用ACK来自此前两步；Attention等当前AIV的KV(r)、此前AIV的PROB(r−2) |
| 同组两个AIV | `AttentionTick_g(r) → ScoreStep_g(r)` | Attention只等此前AIC的SCORE(r−1)、PV(r−3)；随后消费当前Score并ACK |

依据是 `indexer_score_stages.py:174–176,375–378,436–474` 和 `attention_stream_stages.py:141–170,261–391`。AIC在当前Attention前的Score叶尾可能等当前ACK，但AIV当前Attention不等当前AIC的Attention结果，能够先结束并消费该Score，因而没有同轮回边。两角色都采用Score→Attention虽然也可无环，却没有这里的供数交错机制。A3 `wait_flag_dev` 会阻塞后续流水发射；本证明没有把它当成异步等待或假设一个核同时执行两个阶段。

必要边界如下；这些条件不是现有源码已经实现：

- **r不是全24组统一工作量。**各组保留独立Score叶/step游标和Attention query/tick游标，按两者本地总轮数较大者运行；一侧耗尽就只推进另一侧，同组AIC/两AIV必须从同一device描述符决定skip。原owner仍为 `item%24` 与 `q%24`，不能将wave内任务重新编号冒充owner不变。
- **B16也不整齐。**`indexer_group_stages.py:243–249`中groups=16时item=leaf×16+group。连续group0..3的6叶共24任务，只落AIC0..3、8..11、16..19这12核，各2叶；query0..23才是24个Attention组各1条。`ceil(24/max_leaves)=4`能定义wave大小，不能证明填满24核或相同步数。
- **根发布必须晚于根store。**Score ACK只保证FIFO被MTE2读取，随后还有排序及根写回（`indexer_score_stages.py:473,592–602`）。每wave的leaf/root release必须覆盖所有真实叶和尾lane，merge只读已发布的本query根，plan再获取所需TopK。不能把leaf最后ACK当根ready，也不能重复写同一代际ready值表示不同wave；需要独立wave槽或另证单调进度，继续保持512B单写者及原发布/获取维护。
- **plan不能读未来wave。**原plan一次加载M8 TopK（`attention_plan_stages.py:155–159`）。保留其算术时，切点必须由device group offsets找完整group且query尾8对齐的位置；最后T尾块沿原合同处理。混合Q/S1/S5找不到内部合法切点时只能回退单wave，不能只用B或T/6。另一种逐行plan拆法需要另审，本报告没有默认授权。当前wave计算尚未完成的角色不能先阻塞在未来wave的root/plan上。
- **跨wave仍是数据依赖，不新增72核barrier。**本组本轮两段及各自尾部排空完成、根release后，才进入下一wave的实际root→merge→plan依赖。等待可以让快组停住，但慢组尚在处理的数据只依赖已就绪前波和本组FIFO，不反向需要未来plan。保留原QUERY、ORIGINAL_CACHE、MAIN_OUTPUT、ROPE acquire（`cube_schedule.py:381–383`、`vector_schedule.py:890–893`）；不能丢掉旧release链顺带提供的可见性。最终packed发布及O-A获取不提前。
- **空工作与尾部保持配对。**Score的work_valid=0双方都不发事件；有效叶的空query lane仍消费并ACK。Attention双方沿相同mask跳过该块事件，但继续原逻辑tick及query末尾reset。最后wave没有后波Score时仅跑Attention；C仍排空+2拍、V仍排空+3拍，保留Score每叶ACK drain。EARLY_KV的T仍为完整输入T，不能用wave大小改变144阈值。

## 3. 异步存活与两槽替代的边界

Score双槽、Attention三槽和V侧roots/segments、m/l/oi都跨本地轮存活。它们不是每步函数返回就可以覆盖的临时量。独立资源审查见[资源报告](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_attention_interleave_resources_20261018/README.md)及同目录 `API_GAP.md`：原独立Mat驻留至少448+112=560KiB，超过512KiB；Score Left60KiB与Attention Left16KiB有真实地址重叠。UB必要驻留加gather名义168.25KiB低于184KiB预算，但不是planner通过。保留三槽而借死槽需要异dtype子区别名，直接boxed reinterpret已被现有API条件排除，通用MemRef路径尚未证明。

现成 `PAR/oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r/kernels/aic/cube_schedule.cpp` 的tick尾18139–18141发布MTE1→MTE2、M→MTE1、FIX→M局部event0，下一tick17664–17666获取。最后PV TEXTRACT在18077，MMA在18088，TSTORE在18134。借用旧槽必须保留真实pipe依赖，不能按Scalar tick_end判断DMA/数学完成；两个helper现有局部event编号重用也禁止直接拼接生成C++。

**补充：仅AIC Mat KV改两槽，GM所有ring及mi/li仍三槽，改为先PV(r−2)再QK(r)，拓扑上仍无环。**当前PV等上一AIV轮已经发布的PROB(r−2)，其后QK等当前KV(r)；AIV仍只等上一AIC轮SCORE(r−1)和更早PV(r−3)，与上表Score/Attention顺序相容。Mat必须独立用r%2，GM继续r%3；旧PV完成对该Mat槽的MTE1读取后，才允许QK的MTE2写入同槽。首两轮无PV，末尾drain和mask事件配对照旧。该办法避开三槽死槽的异dtype借用要求，但仍须处理Score Left重装、UB重排和联合lowering依赖；改变QK/PV先后会损失原来的QK先行重叠，**这里仅确认有向顺序，不确认资源生成、数学/设备正确性或性能收益**。

独立资源审查已补[两槽容量复核](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/score_attention_interleave_resources_20261018/ALTERNATIVE_TWO_L1_SLOTS.md)：64KiB Q+256KiB KV+48KiB ScoreQ+112KiB Score临时+16KiB PV=496KiB；再保守加入初始化Coeff Mat12KiB为508KiB。Left分时最大60KiB。两槽可沿现有同dtype create/gather/slice表达，仍未证明联合lowering没有额外回边搬运或分配退化。

可继续讨论的是上述有状态交错机制，而不是已失败的整helper排空重排；本轮不据此进入实现或声称能达到30%。源码最小切分/状态表由indexer审查独立负责，资源表达仍需满足其明确边界。
