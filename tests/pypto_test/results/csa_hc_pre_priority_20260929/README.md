# HC前门控优先的最小调度候选

基底是已保留HC_post残差常驻的d93bba14。整包来自该项已验证候选，
共享HC_post同步到生产正文；私有baseline/candidate冻结，只改candidate HC_pre任务前置。
Native最新七档已经收齐，此候选只测PTO长B16/短B24，不重测Native。
沿用现有PTO编译包装（inplace=True），不是要求PTO复刻Native编译选项。

当前55b89ee2七档长B24中，comb首次启动62.050μs，pre/post67.565μs；
两者都依赖linear_reduce，comb不是mix_x_rms_norm前置，不能据同时出现便断言阻塞。
候选将split_pre_post的SPMD改为具名TaskId，把它加到comb_sinkhorn的deps。
算术、token分工、task数量、early标志和数据布局保持；mix仍只等待自己的真实输入。
目标是避免非attention关键链上的Sinkhorn抢先推进，让pre→mix→QKV更早完成。
如果增添前置反而压缩重叠或增加等待，按完整CSA与P95实测否定，不强行保留。

Native最新HcPre在一个融合kernel内完成门控；本接入拆成多个task，需单独验证交接。
上游历史725μs图也有comb比pre/post早的窗口，且其入口为FP32，无本接入BF16 widen；
本候选不是照抄上游优先级，也不将旧图作为同输入严格A/B。先前hc_widen早派发失败，不重复。

CPU完整依赖解析/PTOAS/CCE/load先通过，再正常auto提交一组两档A/B；
5预热20次正式事件、独立各四窗DFX、八类完整状态零容差。
记录HC前段start/end及官方图中实际新增的直接依赖，分别判断核内、CSA、P95，长短8:2。
不增加七档/EP16或无关边界测试；纯依赖变更若有收益，再记录其功能验证范围。

[准备](prepare.py)、[唯一补丁](candidate.patch)、[来源](source.json)、[编译](compile_candidate.json)、
[入口](run.sh)、[收集器](collect.py)。

完整CPU PTOAS/CCE/load、两根依赖图、Ruff/shell均通过。
2026-09-29 10:58正常auto提交`task_20260929_105848_277267824307`，最长5400秒。
任务现已completed(exit=0)，八类完整状态零容差、图重放/保护区通过，16个DFX窗口官方join、行数和block核对通过。
固定window_3证明两档候选均有pre/post→comb直接依赖，baseline没有；预期代码变更确实执行。

**不采用此调度候选。** 长B16完整CSA 999.792→1018.695μs（+1.891%），
短B24 931.264→925.074μs（−0.665%），长短8:2回退1.380%。
长档P95 1008.760→1033.200μs，短档951.020→943.280μs。
各组0/20超过自身P50的105%，仍不代表已关闭历史间歇长尾。

独立DFX中pre/post完成时间长/短分别提前6.530/3.865μs，mix完成提前4.360/3.890μs，
但这不支持正式完整CSA的收益；不将两套独立样本相减推导“核外新增开销”。
Sinkhorn长档核时下降8.028%，短档增加0.602%；本次没有核内算法改动，调度与资源竞争可改变worker计时，
不能把这一局部读数当作应独立保留的核内优化。生产HC_pre保持原实现。
停止边界、全七档和EP16扩测，不在短档单独增加该依赖。
[正式结果及原样本](RESULTS.md)、[交接时间与实际DAG](SCHEDULE.md)。
