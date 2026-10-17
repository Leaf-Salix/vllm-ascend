# 两组 query 完整流水的 CPU 评估（2026-10-17）

当前六个排程原型都没有显示收益，暂不生成对应的设备算子，不提交 NPU。
这不是所有更细流水都不可能优化的证明。

前一轮的所有权模型证明：保持每个 worker 的叶子集合，先完成前8个请求，
可以让其数据在11个score块后就绪；但原整阶段通知仍会等满22块。
因此本轮模型同时拆分 Indexer、merge、plan、QK、O-A、quant、O-B 和 HCpost，
每组48个query。没有只改Indexer后假设下游自然提前。

`query_group_schedule_model.py` 读取现选 B16 调度、已证明的行依赖和同形动态L1成本：

- 产生1313成员、1121调度单元、43953条数学边。
- Indexer与QK保持原logical block到物理核组映射。
- 同一MIX组下一阶段要等前一阶段三成员都结束，保护FFTS复用。
- 同一logical worker的GM交接区在两半之间不得重叠使用。
- 按实际query行重新核对Indexer→merge→plan→QK及投影消费者集合。
- plan原12个有效8行成员分成每半6个；HCpost的48个2行成员分成每半24个。
- 保留现有前端和其他未拆分任务；未假设主Q投影的96行可以提前分批完成。

这里的时间只是成本模型。拆分计算按原成本的一半估算，每个新成员仍保留1μs。
O-A/O-B也先按理想的一半估计，没有把新增权重读开销作为已测值。
因此模型不表示实际性能，不保证任何数值、内存分配或设备同步正确性。

| 排法 | CPU估计总长 μs | 同模型现选 μs |
|---|---:|---:|
| 优先前半组链路 | 855.325 | 771.530 |
| 最早可启动 | 780.470 | 771.530 |
| 剩余关键路径优先 | 813.585 | 771.530 |
| 最早可启动，允许重排MIX物理组 | 780.470 | 771.530 |
| 前缀先排、前半链路优先，允许重排MIX物理组 | 841.340 | 771.530 |
| 仅保护主Q两阶段、再前半优先，允许重排MIX物理组 | 862.105 | 771.530 |

前半优先时Indexer抢先占用资源，主Q投影延后，第一半QK直到384.31才开始；
另两种排法中，第二半QK仍把第一半O-A挤到后面，未形成期望的全链路交错。
这些是对应模型的排队现象，不能替代真实硬件归因。

结果分别保存在 `query_group_schedule_ideal_phase_first.json`、
`query_group_schedule_ideal_earliest.json`、`query_group_schedule_ideal_critical.json`，
选择结论在 `query_group_schedule_assessment.json`。

```bash
python query_group_schedule_model.py --policy phase0_first --projection-fraction 0.5 --output query_group_schedule_ideal_phase_first.json
python query_group_schedule_model.py --policy earliest --projection-fraction 0.5 --output query_group_schedule_ideal_earliest.json
python query_group_schedule_model.py --policy critical --projection-fraction 0.5 --output query_group_schedule_ideal_critical.json
```

没有继续扫描更悲观的投影成本，也没有以“前半数据提前一半”宣称整层会快一半。
如果将来继续此方向，需要能同时处理主Q生产时序、MIX资源复用及输出投影权重开销的新排程依据。

补查的两种排法分别在`query_group_schedule_flexible_earliest.json`和`query_group_schedule_prefix_then_phase0.json`。允许每半不同物理组时仍检查每个MIX调用24组一一对应、每个logical worker交接区的跨半区间不重叠。结果未改善，不据此继续上卡。
