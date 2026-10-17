# QB→DQ→Attention 分块交接：本轮不推进

只读当前正式33源与既有长B16三轮 phase/tick；不改源码、不编译、不上卡。结论是**没有证据支持把这两处交接改成下一项实现候选**：当前DQ入口已晚于全部QB完成，而Attention的plan获取链已经包含全部QUERY完成。已有全量GM中间结果确实较大，但只拆通知不减少搬运；真正消除它需要改变跨核所有权和长时间片上驻留，尚无可直接复用的安全实现或关键路径依据。

当前正式DQ已包含9d分组；T96仍走原16-head分组。因此下面d253/T96诊断能用于这两个未变数学路径的局部判断，不能外推所有T、所有控制策略，也不把插桩总时差当完整L1收益。

## 时序：当前没有等QB生产的入口门槛

数据来自 `parallel_20261018/attention_tick_h131072_b16_v1/clock_round_{0,1,2}_spmd_reference/phase_clock.json`。精确路径、大小、统计和计时限制在 [evidence.json](./evidence.json)。按源码实际前驱，逐AIV取 `IDX_SCALE.notify_end→QUERY.notify_start`，不依赖phase编号排序。

| 同轮端点/μs | 第0轮 | 第1轮 | 第2轮 |
| --- | ---: | ---: | ---: |
| 全部QB最晚发布结束 | 261.06 | 255.66 | 253.64 |
| 最早IDX_SCALE发布结束，即DQ等待前驱下界 | 273.78 | 268.92 | 268.28 |
| 最早DQ入口晚于全QB就绪 | 12.72 | 13.26 | 14.64 |
| DQ区间逐AIV中位 | 22.36 | 23.21 | 24.30 |
| 其中24次QB信号检查累计中位 | 3.37 | 3.08 | 3.24 |
| 同轮DQ区间最大值 | 23.82 | 24.56 | 25.70 |
| 全QUERY结束至最早任意plan发布开始 | 266.84 | 267.60 | 266.58 |

`vector_schedule.py:715–768`先完成COEFF、compact量化、scale页提交，再等QB和执行DQ。故在现有顺序下把QB按列/head更早发布，不能让AIV更早进入DQ；3.08–3.37μs是已就绪信号的检查区间，不能解读为仍在等待QB生产。phase计时还包含原发布cache维护与插桩，DQ区间包含获取和数学/搬运，不能称纯计算或纯DMA。

量级上，即使把整个DQ区间凭空删除，固定其它工作成本时，每个AIV局部推进预算也仅23.82–25.70μs；这是极不现实的局部上限，包含不能删除的数学，不是整层收益预测。单纯QB分块交接在当前入口没有可省的生产等待，信号检查节省也不能按次数预记收益。

## 唯一具体反例：按QB列轮次发布仍没有消费窗口

实际生产在 `query_projection_stages.py:_q_b_nz_stage`：NZ为M128/N256/K256，列块n=0..127归 `n%24`，轮次 `n//24`；同一owner先完成该列的全部动态M128行块，再做下一列。`cube_schedule.py:311–312`目前只在整个helper后发布PHASE_QB。

实际DQ在 `query_projection_stages.py:q_dequant_rope_stage`：一个head需要两个完整N256输出，head h的生产者是 `(2*h)%24`、`(2*h+1)%24`。DQ任务owner为 `((row//8)*(64/head_tile)+head//head_tile)%48`。T96/head_tile16时，一个任务需要32个连续列块，已覆盖全部24个QB物理owner，不能仅把现有24次等待缩为“两个owner”。

一个可静态定义、没有新增全核barrier的列轮次协议，是新增6个轮次bank，每个AIC只写自己的512B槽；完成本轮该列的所有动态行后以原generation发布。DQ按所需列集合，为每个owner等待最大所需轮次。最后一轮只有owner0..7有效，消费者不等无生产列；跨调用不清旧槽，仍用generation排除旧值。AIC不等DQ，因此新增边本身无环。

但该协议保留全部q_proj GM，**业务GM字节节省为0，新增UB为0**。若沿用72核phase布局需额外6×72×512=216KiB信号空间（专门24-AIC布局为72KiB）；发布从24增至128，每次仍须保留业务排空/cache维护。最关键的是DQ要等前面的IDX_SCALE，本表证明它此时已可取得所有原QB完成槽：新增104次发布没有当前消费窗口。本轮据此否决这项分块方案，不写候选。

## DQ→Attention：全局QUERY等待已在plan的获取闭包内

`vector_schedule.py:768`的QUERY发布在每个AIV自己的SCORE_VECTOR之前；`vector_schedule.py:825–864`在TopK前获取全部48个SCORE_VECTOR，之后才产plan。AIC在 `cube_schedule.py:379–381`先获取实际plan owner，再等全QUERY。因此plan包含全QUERY的release/acquire传递；即使行/head细分QUERY，现有Attention仍必须先得到本层TopK/plan。该结论来自源码偏序，不依赖表中约267μs的时间间隔。

单独折叠QUERY等待已在 [sync_evidence](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/sync_evidence/README.md) 的实际候选中做过：长B16回退0.40%、短B32改善0.31%，不采用。此处不是新的全局门槛，不再重复该试验。要让Attention与未完成Score交叠，必须连同Score、TopK、plan和FFTS角色切换一起改；固定owner的two-wave模型已被否决，不能仅换QUERY通知包装重做。

## GM与驻留边界，不把字节数当时间

T96的有效逻辑数据为QB INT32输出12MiB、DQ读取12MiB；DQ BF16输出6MiB、Attention query读取6MiB。后者已在每个query首块载入一次L1，随后五个候选块复用同一qk_q，见 `attention_stream_stages.py:134–151`，不是每块重复读query。上述是有效数据载荷，不能等同实测HBM流量或按带宽推断耗时。

现有AIC列分工让一整个512列head来自两个不同AIC，DQ又按八行/head组分给AIV；普通通知只能使GM可见，不能直接让一个AIV访问两个AIC的私有Acc。若保留整个M128/head输出等待DQ，需要256KiB INT32，而当前单个Acc容量128KiB；若改成同owner或更小M并保留完整head，已进入Q_B数学分块与权重重读问题，不能称只改交接。

另一种直接保留DQ输出到Attention的想法，在T96下平均每AIV需要128KiB BF16持久数据，并且要跨过主压缩、Score、TopK。沿用当前DQ最高地址末端128KiB而另开独立持久区，会把末端推至256KiB；复用地址需要新的全程存活期与所有权证明，现有生成物并未提供这份余量。这里只说明额外资源和别名证明义务，不把地址末端冒充实测live峰值，也不宣称任何重排都绝对不可能。

## 历史排重与停止边界

已查MASTER、`ROW_DEPENDENCY_EXPERIMENT.md`、`QUERY_GROUP_SCHEDULE_FEASIBILITY.md`，以及主PAR中的 `sync_evidence`、`wave_audit`、`qb_tile_readonly`、`query_head_group_evidence`、`query_linear_evidence`。旧行依赖细化主要改变merge/plan/QK/OA成员边，并未证明当前QB入口在等数据；旧主Q优先级/两组query排程没有形成收益；QB Left双槽实际回退0.657%，N512只读审查又指出Right容量与权重重读代价。未用这些不同试验冒充“同一个列轮次通知已测过”，本轮否决来自当前消费时刻及明确资源/所有权障碍。

到此收口。没有新的可信分块候选，也不申请诊断或模型扩展；当前证据不足以从这两处交接找到接近逐档30%目标的改动。
