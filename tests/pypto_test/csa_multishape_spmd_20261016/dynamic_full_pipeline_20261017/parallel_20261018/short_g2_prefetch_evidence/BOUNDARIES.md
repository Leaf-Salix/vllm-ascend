# PREFETCH 新区间的 S1/S5/live 尾部边界（只读建议）

本次只改 G2 短档的 Key 搬运时机；QK M128/K128、WS K128、FP16 转换、半叶384、
根数与排序输入次序不变。S1每组live1，S5为live2+2+1，原 A3 尾组补齐修正均走同一
正式 helper：GM只读live*64行，在L1重用本组首条有效query补齐物理M，再搬至Left。
新的Key池在Query/coeff搬到Left后复用其L1空间。根据源和生成物没有新增有效query
读取，也没有改变实际可见候选范围；这不是设备 exact 的替代证明。

第p个panel的消费地址仍为 `leaf_begin+step*384+(p//3)*lane_span+(p%3)*128`。
预取只把这个地址的提交提前到前一panel，prime补上panel0，最后不读取panel6。
因此访问的逻辑页集合相同，仍通过cache长度clamp与Key0/Key64视图访问；末尾无效
score仍由各query的visibility在AIV裁掉。短G6与全部长档策略未改，不建议复测这些分支。

既有尾组探针 `dynamic_full_pipeline_20261017/score_tail_probe.py:86` 调用
`dynamic_indexer_score_20261017/cpu_probes.py:217` 的 `g2_768_probe`，其中参数明确为
G2/N128/768/**PREFETCH0**/TO_L0true。因此旧live2→1→2与整链S1/S5验证证明了原尾组
修正，却不自动覆盖这次新Key池/Right预取的生命周期。此前150轮score缓存helper也
不能当作PREFETCH1的验证，更不应原样再跑150轮。

若两个S6完整矩阵均通过，最小有价值的追加是一个同进程旧/候选Score helper对照，
复用已有输入、精确oracle、同址图和保护区合同，只取以下三个visibility：

| visibility | 新边界价值 |
| --- | --- |
| 2048 | S1/较早query的完整压缩历史，尾lane的实际有效数少于384 |
| 2049 | S5最后live1 query可跨ratio4边界，新增一条候选位于末页 |
| 8192 | 当前短区间上缘，lane_span被4096封顶，最终step只有256有效列而非384 |

每点只做live2→1→2，合计9个同址状态，比较原有每个有效query的完整score、transfer
和两个Top512根及只读cache/未写区。它针对新的预取生命周期与返回完整组后的复用，
不需要重新做全部六个TopK分支/全同分150轮，也不需要四档完整S1/S5因果参照。
原sort/mrgsort、tie顺序、组/叶映射均没有修改；上述建议尚未执行或提交。

若只核对本轮8K两档的S1/S5，可以只取前两个点共6个状态；若要采用整个短G2区间，
再包含8192点的封顶行为更有价值。具体是否追加由主 agent 根据整链收益决定。
