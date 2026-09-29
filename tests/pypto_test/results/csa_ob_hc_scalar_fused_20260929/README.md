# 整D单行标量收尾融合

基底仍为9a868d26对应的已测O-B优化，使用新的baseline/candidate私有整包。
前一T16/N512、内部T8融合长B16总核内工作量+144.494%、CSA+1.173%，已否定，
见[完整长档失败证据](../csa_ob_hc_fused_20260929/PARTIAL_RESULTS.md)。
失败版同时引入了HC多行广播和转置，不能只因GM交接减少就认为融合有效。

本轮恢复HC原有的每worker最多四token、单token整D4096向量及标量门控：
任务T4/N4096，内部T1；长B16为24份worker，短B24为36份。
每token八组INT32部分和依次按FP32 scale相乘并相加，后乘权重scale、RINT为BF16；
随后直接在同一UB内执行原HC单行算术，仍保留BF16→FP32和最终BF16 RINT。
反量化的每组scale也改为标量读取/乘法，与逐元素组归约顺序相同。
共享helper只保留原单行实现，残差四路仍一次加载后复用，无新增算术策略。

相对上游pypto-lib，按组量化和逐组相加继续保留；区别是将Native BF16输入/输出要求下的
收尾交接合并，去掉attn_out GM缓冲和独立HC任务。HC残差常驻仍源于已采用的
最新ops-transformer Permanent-X参考。此次跨算子边界融合不冒称Native现有实现已有同款融合。
还改变了反量化分块和worker数，后续必须按总worker核时及区间评估，不能只比单worker均值。

两入口依赖解析、完整PTOAS/CCE/link/load和共享HC单行入口CPU编译通过。
三条行块特化的Vec上界均131072字节；生成码零TROWEXPANDMUL、零TTRANS、零TMOV，
四次最终TSTORE，中间BF16 RINT→FP32仍存在；无attn_out GM分配。
八组O-B显式依赖、post/comb输入自动依赖仍保留，设备DFX会逐项验证。
Ruff和shell语法通过，没有修改工具链或生产源码。

14:14通过正常auto队列提交task_20260929_141413_142108726634，设备1上已退出0。
两份Python源码已设只读。128K/B16先baseline后candidate，8K/B24反序；
固定CANN9.2/mode2/atomic0/det0、第二个CSA层权重和合成历史，
5预热20次正式计时、四窗DFX、八类完整状态零容差；分别报告CSA/P95、总核内工作量和跨度，长短8:2。
两档八类跨版本状态零容差、自身图/保护区、16窗官方raw join和worker覆盖均通过。
长档CSA984.188→973.438μs（−1.092%），短档934.461→930.869μs（−0.384%）；
P95分别992.080→985.060、950.560→943.900μs，四组均0/20超过自身P50的105%。
收尾总核内工作量长档757.710→674.590μs（−10.970%）、短档1323.150→1208.805μs（−8.642%）；
两档四窗范围完全分离。收尾跨度分别39.140→29.655和44.920→35.490μs。
长短8:2为核内−10.504%、完整CSA−0.951%，不是Native或模型对比。

真实融合任务仍依赖八组O-B及post/comb，最后前置FIN→首start为长档0.66–0.82μs、
短档0.58–0.86μs。基线同样小于1μs；收益不能写成解决原本不存在的长ready等待。
短档未改O-A和Sparse部分窗口更慢，不抹去或直接归因于此改动；
独立DFX与正式CSA不做相减。核内收益和减少一级任务/GM交接共同保留为结构变化。
与首版的对照说明恢复单行分工有效，但没有隔离实验逐一量化标量乘、分块与融合各自贡献。

按核内及8:2判据值得保留；14:29提交边界task_20260929_142920_279636119594，
设备0上已退出0。只测H127/B3/T18、active-B=3/2/1/3固定图padding；
另测共享HC单行入口T18的eager/graph/更新输入，对CPU逐项算术及跨版本均零容差，带输出保护区。
该入口抽取了公共helper，需直接覆盖，不能用融合路径的PASS替代；不扩大到Native/七档/EP16。
边界八类跨版本状态、同图padding、保护区全部通过；共享HC三种调用对CPU参考和跨版本均零差异。
核对生产与冻结baseline一致后，已把candidate的O投影、CSA调用、共享HC三个文件移入生产，
仅补两处说明性注释。性能包原有重导出已覆盖新helper，无需修改第四个文件。
两版两个入口的生产依赖解析及Ruff通过；精度版仍独立调用共享HC，没有迁移性能版的组量化。
本次只采用已测结构，不重复跑同两档或Native，不将本版两档覆盖称为完整新七档/EP16验收。

[正式结果](RESULTS.md)、[样本和完整状态](summary.json)、[官方join与全部任务](evidence.json)、
[四窗范围及未改任务](decision.json)、[边界入口](run_boundary.sh)、
[共享HC检查](shared_hc_case.py)、[边界收集](collect_boundary.py)。
[边界与共享HC结果](boundary/summary.json)、[生产两版入口解析](production_parse.json)。

[源码变换](prepare.py)、[差异](candidate.patch)、[来源](source.json)、
[CPU静态证据](static_evidence.json)、[CSA编译](compile_candidate.json)、
[共享HC编译](compile_shared_hc.json)、[设备入口](run.sh)、[收集](collect.py)。
