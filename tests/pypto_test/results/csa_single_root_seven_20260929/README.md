# 单根Indexer采用后的七档缺口验证

采用提交2ed8ae2e；生产性能版与已测私有候选代码体一致，精度版未迁移。
完整两档A/B及B4/H65535尾段/padding已通过，[依据](../csa_score_single_root_20260929/README.md)。

为避免重复占卡，复用同一冻结源码已完成的128K/B16、8K/B24，只补：
128K B4/B8/B24，以及8K B16/B32。全部沿同一私有pkg，不按档位切历史源码。
每档5预热20次正式设备事件，独立PyTorch profile和四窗level-4泳道，保留原样本及P95/max。
Native沿用csa_native_inplace_seven_20260929的七档，CANN9.2、npugraph_ex、dynamic=False、
inplace_pass/static/SuperKernel开启；核内参考沿已有static开/SuperKernel关profile，不新增开关测试。

这是一份当前已有Native与同源码PTO的汇总，采样分属不同任务，逐档记录来源和设备；
不能冒称同次A/B或用它推导单根策略的因果收益。该项因果对照仍是上面的两档同卡实验。
同一套源码的分支覆盖与七档核内范围需由真实结果确认，不凭“短档代码没变”填补数据。

收集器检查七档各自图重放的八类状态、Top-K结构和保护区。
旧七档没有保存完整state，不能声称七档跨版本零差异，也不为补存快照重跑旧版本。
跨版本逐元素依据仍限于已完成的两档A/B和尾段/padding；不能替代Native精度或真实EP16 token/DSpark验收。

七档PTO固定window_3的原始merged泳道已汇集到download_pto_swimlanes，
按01_128K_B4至07_8K_B32编号，文件名包含PTO_Swimlane、算子版本与SingleCSA_SyntheticHistory。
不改事件，不选最快窗口；全部四窗和独立PyTorch profile仍留原目录，并在来源表中列出。
同时生成一个ZIP便于一次下载；原始事件保持不变。

任务task_20260929_120323_390116317823已完成（exit=0），auto设备0已释放。
五档补测加两档同源码复用全部收齐；28个DFX窗口官方join/行数/block核对、
七档自身图重放状态/Top-K结构/保护区通过。Native与PTO均在设备0，但采样轮次不同。
对现有Native：128K均值变化−10.590%、8K +2.667%、8:2 −7.938%；短档仍慢，结果不隐藏。
PTO各档0/20超过自身P50的105%；本轮不关闭历史间歇尾部或EP16问题。

[正式结果](RESULTS.md)、[任务明细](TASKS.md)、[精简证据](summary.json)、
[七份泳道目录](download_pto_swimlanes/README.md)、[ZIP下载](PTO_CSA_7cases_2ed8ae2e_20260929.zip)。

[入口](run.sh)、[收集](collect.py)、[七档打包](bundle.py)、[来源](source.json)。
