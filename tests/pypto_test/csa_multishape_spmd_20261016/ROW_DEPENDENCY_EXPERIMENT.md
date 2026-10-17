# B16/S6 单 SPMD：按 query 与 head 细化成员依赖

2026-10-17。实现位于 `row_dependencies.py`，仅允许长历史 B16/S6、独立临时区、
原数学体与原核内顺序，可组合此前已选的 QK 地址缓存。其他 batch 不直接复用此等待表。

## 改了什么

原实现把某个 task 的输出作为整体建立依赖。本候选在核对每个消费者实际读取的行后，
将四类完整 task 依赖改成对应的成员依赖。这确实放宽了原 task 的数学偏序，
不能宣称“原 task DAG 完全不变”；重新验证的是细化后依赖图、核内顺序与信号翻译的一致性。

| 消费者 | 实际需要等待的生产者 | 依据与边界 |
| --- | --- | --- |
| Top-K merge | 3 个相关 Indexer 核组的全部 9 个成员 | 长历史 T96 的 item 按 leaf-major 排列；merge worker w 处理 q=w,w+48；覆盖所有 1～32 个 leaf 的可能生产者 |
| 有效块计划 | 所处理 8 个 query 的 merge 成员 | plan worker w 处理从 w×8 起、步长 128 的行块；T96 时 12～15 号 plan 不读 Top-K |
| QK/PV 核组 | 本组 4 个 query 的 plan 成员 | q=c,c+24,c+48,c+72；其余 Q/KV/RoPE 父任务依赖保留 |
| O-A 投影 | 对应 head 组的 24 个 AIV 成员 | AIV0 写 head 组 0～3，AIV1 写 4～7；packed 行区间不重叠 |

另为每个 QK 成员保留同一物理核组前一 Indexer 的 AIC+AIV0+AIV1 完成依赖，
共 216 条，用于约束 FFTS 资源的阶段复用。这些顺序本来就包含在原图祖先关系中。
每核原有执行序列、GM 独立分配、pipeline fence、DCCI/DSB 与通知发布不改。
“只等数据所属核”不能成为删除资源复用依赖或缓存发布的理由。

## CPU 证据

`row_dependencies_prepare_v1/control_variants/h131072_b16/row_dependencies/` 保存
`audit.json` 与 `member_graph.json`。关键循环与存储下标有 AST 契约检查；
输入 Tensor 的 allocation/offset/shape/stride/dtype/layout 必须吻合，其他共享引用
只允许原只读根输入，不能隐含共享临时区。

- 1536 个 merge worker×leaf_count 组合（每个 worker 两个 query）覆盖 96 个 query、1～32 个 leaf，及全部 plan/QK/packed 行。
- 逐核远端进度等待项：16,729 → 12,183；发布点：730 → 722。
- 原成员边 40,155；细化后必需边 32,805；计划边 17,630 → 13,084。
- 传递约简及发布序号翻译后，逐节点祖先集合与细化后的依赖图相同。
- 生成 `csa_static_aic/aiv/group/entry` 的函数体 AST 与此前候选一致。
  变化集中在等待表与发布表；参数的等待表长度描述相应变化。
- 原协议模型的多代检查通过。它不代替 NPU 内存一致性或数值验证。

`row_dependency_cost_model.json` 使用旧动态 PTO 每成员 L1 成本估算最长依赖链，
748.55 → 748.53 μs。此数不是实测，也不包含新的等待开销；它提示本次改动主要减少
等待表开销，不能指望仅靠删这些依赖就消除约百微秒的计算路径。

## 首轮完整设备结果

任务 `task_20261017_044912_40225339550`，目录 `b16_cached_row_deps_v1/`，exit=0。
正式第 4 层 W8A8 权重，原动态 PTO 与单 SPMD 输出、Top-K、六份 state/cache、
eager/graph/最终结果逐 bit 一致；30 代完成信号、只读区、未写区及工作区保护区通过。
两侧均为 PTO，没有执行 Native forward。

| 完整无插桩 L1 incore 首尾 | 三次采样（μs） | 中位数（μs） |
| --- | --- | --- |
| 原动态 PTO | 888.48 / 905.08 / 909.80 | 905.08 |
| 行依赖单 SPMD | 807.16 / 804.40 / 811.26 | 807.16 |

相对同轮动态 PTO 缩短 10.82%。此前所选单 SPMD 的历史中位数为 815.84 μs，
只相差约 1%，不能把跨轮差值直接当成稳定收益。尚未达到 700 μs。

## 同进程复核与动态长度

`paired_row_dependency_perf.py` 在同进程、同权重及根输入/输出地址下，
分别编译此前所选单 SPMD 和行依赖候选；各采 3 个完整 L1 窗口，交替先后顺序。
两边的工作区和进度信号独立，分别验证输出与信号代际。该辅助检查只由
`--paired-selected-spmd --row-dependencies --metadata-replay-only` 显式启用。

同一任务随后保持候选图、地址和二进制，执行
131072 → 131071 → 65536 → 131072，并交换页表、更新位置与 RoPE。
64K 使 Indexer 从 6 个 leaf 变为 3 个，覆盖本次等待表依赖运行时 leaf 分工的风险。
CPU 编排签名确认这些长度仍使用同一数学策略；16384 跨策略，明确不放入该图。
证据为 `row_dependency_metadata_strategy.json`。

任务 `task_20261017_050033_181823706`，结果目录 `b16_row_deps_paired_metadata_v1/`。
此段先记录验证范围；终态与采用结论以下文补记为准，不预记 PASS。

### 终态与采用结论

任务 exit=0、完整 PASS。旧单 SPMD 三窗为 841.94/843.14/833.20 μs，
新候选为 806.84/817.88/821.56 μs，中位数 841.94→817.88 μs，缩短 2.86%。
新候选最慢窗口也快于旧候选最快窗口；在本轮范围内收益得到确认。
不能把首轮 807.16 和复核 817.88 当成同一轮的不同统计量。

A/B/C1/A_return 四个变体全部通过原动态 eager、原动态图与候选图的输出/Top-K/
六份状态及保护区检查。固定地址、无新增编译单元通过；候选共11代、旧候选7代
进度检查通过。是不同输入快照重放，不是连续decode状态轨迹，也没有覆盖prefix共享。

已更新 `selected_shape_policies.json`：只为128K/B16选择行依赖细化；
其余六档仍使用各自已有候选。本目录为实验实现，未改生产入口。
