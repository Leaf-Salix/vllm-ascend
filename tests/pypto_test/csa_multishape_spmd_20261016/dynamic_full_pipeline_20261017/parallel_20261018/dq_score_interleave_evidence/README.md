# 主 Query 与 Score 穿插：优先处理整阶段串行

2026-10-18，基于正式 `f222a917`；候选直接在 ops/pypto 动态源码中实现。
两档已完整执行，未获得足够收益，正式五文件已恢复f222a917，候选完整保留。
静态历史只作参考，没有新测静态，也没有在测试产物上改冻结排程。

## 依据及范围

已有 `dq_score_current_readiness_20261018` 证据显示，协同前段调整后主压缩
先完成，Indexer scale 发布到主 Query 发布的同核区间中位仍约21μs。
该区间含QB获取、缓存维护和插桩，不能把21μs当作保证可省的性能。
源码能直接确认：主 Query 反量化并非 Score 输入，却在所有AIV进入Score
前顺序执行；Score双槽ACK会使迟到消费者限制AIC推进。

历史静态B16参考781.94μs、短B32参考905.72μs；本轮前最近动态分别
827.06μs和943.34μs，均跨轮，不能用相减值归因。这两个代表点用于评估
改变整条推进链是否有收益，不再追加统计环等小候选的重复采样。

## 单一候选

- 将原DQ的八行/head数学提取为共享 `q_dequant_rope_block_stage`，完整行
  保留rsqrt，尾行保留recip(sqrt)，不改512列归约或两次FP32乘法顺序。
- `q_dequant_head_group_size` 保留原动态4/8/16/32/64分组与owner。
  Scalar进度为原 `work=worker+k*48` 及本任务内head_offset，跨Score叶携带。
- balanced偶数步、非balanced每步在原ACK及本步scale/segment写入后，
  至多推进8head。每次让回Score，不在已有FIFO槽中等待新前驱。
- 所有QB/QR/RoPE获取仍在Score之前。无Score工作或窗口不足的owner在
  本核Score后补齐原DQ任务，再发布QUERY；Attention继续按原QUERY获取。
  不新增全核屏障、GM调度队列、atomic或Host metadata。
- 原Score READY/ACK/drain、Top-K排序与数学保持。没有提前发布QUERY，
  也没有以SCORE_CUBE整段完成作为AIV前驱。

相关正式文件：query_projection_stages.py、indexer_score_stages.py、
vector_schedule.py、entry.py、policy.py。新开关 `INTERLEAVE_QUERY_DQ`
仅为候选消融，默认False；是否进入区间策略取决于完整精度和完整L1结果。

## 当前检查与设备任务

长G6、短G2两类完整符号编译通过，保留58动态轴、130根和一次MIX。
物理分配记录在 `CPU_ALLOCATION_AUDIT.json`，声明末端不等于live峰值。
新编译器沿用独立PyPTO `d01de8523` 及原本地内存复用修改；本轮没有再改
PyPTO/Simpler/PTOAS，没有更改共享安装。

`audit_owners.py` 执行真实源码的Scalar控制部分，Tile数学仅替换为记录桩，
核对T1..512、48owner、0/1/2/3/8/32可用窗口共147456组合。逐owner的
原始有序head覆盖完全相同，包含余量/零窗口和尾行固定4head；Score的
FFTS调用源码保持。此检查不是数学精度、DMA时序或性能证据。

两组正式W8A8层4设备任务均已exit0：128K/B16、8K/B32。具体命令及ID在
`DEVICE_TASKS.json`；同进程原多task/此次正式源/候选五窗完整L1，先执行
精确、状态、A→B→A、图和保护检查，全部通过。
`source_reference/` 和 `source_candidate_v1/` 是实际被测完整源，冻结保留。

## 完整结果与收口

| 历史/B | 原多task | 正式动态 | 穿插候选 | 相对正式降低 |
|---|---:|---:|---:|---:|
| 128K/B16 | 870.08 | 827.12 | 829.22 | -0.254% |
| 8K/B32 | 1003.26 | 954.82 | 942.34 | +1.307% |

全部五窗口保留在 `TERMINAL_RESULTS.json` 与原始source_matrix中。长B16
参考最后一窗为1297.10μs，没有删除它；中位仍827.12μs，其他四窗中候选
三窗更慢，不能用异常一窗宣称收益。短B32五对窗口均改善，但整层约1.31%。
两类实际Vec声明地址末端仍163872B；长B16 AIV bin从146972增到159200字节，
AIC bin均51300字节。代码大小变化不等于已经证明I-cache或回退根因。

本轮说明：把完整DQ拆进Score并未在长B16获得净收益。不能把先前约21μs
区间直接视作可省耗时，也不能仅凭源码交错宣布隐藏了这些计算。为了单点
1.31%引入较大Score耦合不作为当前默认策略，不扩七档、不扫head块大小。
正式恢复经过执行AST和原始源核对，见 `FORMAL_RESTORE.json`；共享依赖未改。

后续保持抓大放小，先核对已有端点里QB实际就绪与Indexer工作窗口的关系。
旧的QB全K常驻、直接GM→L1等完整负例仍须排重；仅有逻辑字节节省不是
重试理由。当前没有新候选或新的收益结论，不把本段写成已经完成后续优化。
