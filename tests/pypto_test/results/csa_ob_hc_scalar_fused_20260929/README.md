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

14:14通过正常auto队列提交task_20260929_141413_142108726634，已确认设备1上running。
两份Python源码已设只读。128K/B16先baseline后candidate，8K/B24反序；
固定CANN9.2/mode2/atomic0/det0、第二个CSA层权重和合成历史，
5预热20次正式计时、四窗DFX、八类完整状态零容差；分别报告CSA/P95、总核内工作量和跨度，长短8:2。
若长档再次明确无核内收益，则停止剩余扩测；有收益再补尾行/padding及共享HC入口。
尚无本版设备或采用结论，前版的状态PASS不替代本版验证。

[源码变换](prepare.py)、[差异](candidate.patch)、[来源](source.json)、
[CPU静态证据](static_evidence.json)、[CSA编译](compile_candidate.json)、
[共享HC编译](compile_shared_hc.json)、[设备入口](run.sh)、[收集](collect.py)。
