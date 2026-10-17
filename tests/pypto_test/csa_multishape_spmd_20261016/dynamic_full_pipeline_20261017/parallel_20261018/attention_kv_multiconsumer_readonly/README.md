# Attention KV 跨消费者复用：66882634 只读核对

结论：**不推进“一组 AIV 为同 query 的多个 AIC 复用 KV”候选。当前没有这类重复采集。** 每个 query 的全部 64 heads 已在一个 AIC 上计算，双 AIV 按候选行分半采集一次，QK/PV 还复用同一份三槽 L1。跨 query 可能命中同一物理 KV 行，但本轮没有地址重叠证据；不能把这种可能性改写成已发现可消除的重复搬运。本轮只读，没有源码修改、编译、NPU 或新性能数据。

## 当前 owner 与 FIFO 数据流

已将当前四份相关源码逐字核对 `66882634`，结果见 `evidence.json`。`attention_stream_stages.py` 与既有 d253 tick 诊断的源码快照也逐字相同，因此下面可引用当时的阶段结构，但不把 d253 插桩时间称为当前整层性能。

| 项目 | 实际分配 |
| --- | --- |
| query t | 唯一组 g=t%24，本组序号 floor(t/24) |
| block | 0=SWA，1～4=512 个压缩候选的四个 128 行块 |
| 本组工作序号 | w=5*floor(t/24)+block |
| GM ring slot | 3*g+w%3，跨 query 连续轮转 |
| AIC g | 同时处理全部 64 heads，QK 为 64×512 乘 512×128 |
| AIV lane0/lane1 | gather 分别处理块内 0～63 / 64～127 候选，各自搬完整 512 列 |
| AIV softmax/merge | 同两 lane 分别处理 head 0～31 / 32～63；这与 gather 的候选轴分工不同 |

关键位置：`attention_stream_stages.py:121–203` 的 AIC、`:233–323` 的 AIV；两侧编排调用在 `cube_schedule.py:384`、`vector_schedule.py:896`。这里的 24 个组是 Attention 的物理 AIC 组；Indexer 的 G2/G6、leaf/root/score 分片不会让 Attention 对同 query 再执行一份 gather。

每个有效压缩候选索引位置只被一个 lane 的一个 row 迭代读取一次。lane 的 32-head 分工不会再触发第二次 KV gather；同一物理行若在输入页映射/TopK 中重复出现，属于数据本身的重复引用，不能与 head 分工重复混为一谈。现有代码保持每个逻辑候选的位置、bias 和乘法重数，不应先做去重排序改变数学。

已完成的复用是：Native cache → AIV UB → GM `kv_transfer` → AIC 三槽 L1；QK 用该 L1 Tile 的转置 view，PV 用同一 Tile 的行/列片段，**不会为 PV 再从 GM 载入一次 KV**。单有效块为 128×512×2=128 KiB，两 AIV 各 64 KiB；一个 query 四个全有效压缩块共 512 KiB 的源数据逻辑搬运，绝不是 64 heads×512 KiB。这个字节模型不意味着每次均访问 HBM，也不表示可直接省去相同时间。

## 多消费者不是现有协议的免费扩展

当前是每组两 AIV 生产一份 128 行块给本组一个 AIC。两 lane 的 MTE3 KV_READY 就绪后，AIC MTE2 才取完整块；AIC SCORE_READY、两 AIV PROB_READY、AIC PV_READY 和 tick−1/−2/−3 的消费顺序共同约束三槽复用与排空。当前没有独立的“任意多个 AIC 都已经取走本 GM 槽”的 ACK 计数。阶段级 phase 发布也不能承担这项逐槽责任。

如果为多个 query 共享一份采集结果，至少要满足：

1. 共享的是同次调用、同物理页版本、同一有序 128 行列表；还要保留各 query 自己的可见范围、mask/bias。只有部分重叠时，需要构造 union 加每 query 映射，或重新打包各自顺序，不能直接广播原块。
2. READY 必须被全部目标 AIC 明确消费；现有组内 FFTS 不能不经新协议验证就当作跨任意组广播。每个消费者应在本次 GM→本核 L1 的 DMA 真正完成后发布对应代际 ACK，生产者等全部 ACK 后才能复用 GM 槽；不能拿 Scalar 指令已发射代替完成。
3. 每个 AIC 仍需自己的 Q、L1、概率和 PV 状态，原本同一 query 的 QK/PV 次序不变。共享 GM 块并不消除多个 AIC 的 GM→各自 L1 搬运，也不能从另一核直接借用这份本地 Mat Tile。
4. 增加的是动态消费者集合、各槽代际/有效谓词、末尾 drain 和慢消费者背压。若用 GM flag，A3 还需正确的发布与读侧缓存维护，不能依赖 Scalar cache 一致性；不得未经验证地把一对原 FFTS 改成多个消费者共同等待。

当前 AIC Mat 的主要对象已是 KV 三槽 384 KiB + Q 64 KiB + probability 16 KiB = 464 KiB，即 475136 B 的上述对象容量；本轮没有编译新的全程序分配结果。AIV gather 半块为 64 KiB，还需在线左右 oi 共 64 KiB及 softmax/其它临时区。24 组现有 KV GM ring 是 9 MiB。新共享方案不一定必须扩大总 ring，但需要重新分配其所有权和生命周期；若再保留额外块/多个 query 的活跃 Tile，会增加 UB/L1/GM 驻留，不能写成无成本复用。

如果人为把同 query 的 heads 拆到多个 AIC，便会先制造当前不存在的消费者复制：每个 AIC 都需要同一 KV 的本地 L1 副本，还减少可并行 query 的核组数。没有证据支持为这个方向做 CPU 候选。

## 现有 tick 与历史排重

既有 `parallel_20261018/attention_tick_clock_evidence/ANALYSIS.md` 表明：128K/B16/S6 三轮的压缩 gather 逐核累计中位为 64.76/62.89/63.80 µs，约占 gather 的 95%；两 lane 快慢随轮次反转。首压缩半块约 6.2 µs，后续块约 2.6～3.9 µs。它没有记录物理地址、相邻 query TopK 重叠率或 cache 事件，不能据此推断大量跨 query 可复用，也不能将这些区间当可节省的整层时间。记录是 Scalar marker 的采集/发射范围，仍含插桩扰动和原有依赖。

已排除的同类方向保持关闭：

- `QK_DIRECT_L1_EXPERIMENT.md`：两 AIV 产物理地址、AIC 直接从 Native cache 逐行搬至 L1，去掉原 UB/GM 中转并新增复用 ACK；完整 exact/图/状态通过，但同轮原动态 892.66 µs、候选 1100.10 µs，不采用。这是改变搬运路径，不是跨 query 共享；不能重新命名后重试。
- 当前 A3 `aic_gather` / TPUSH / TPOP 仍经 GM FIFO，已在 MASTER 的“当前 A3 数据路径纠正”记录核实。换 API 不会自动消除 GM 往返，也不自动提供任意多 AIC 的 multicast/ACK 合同。
- `qk_mgather_result.json` 与 MASTER 保留 MGATHER 整链通过但无收益的旧结果；该方案改变候选搬运粒度，不是发现 head 组重复采集。本轮没有重开它。
- 本轮物理 row plan、重复 guard 删除等已测路径也不重提；它们减少地址工作，未改变此处的单 query / 单 AIC 所有权。

当前可证实的新结论就是“同 query/head/score 分片重复采集”这一前提不成立。跨 query/prefix 复用必须先有真实有序候选及物理地址重叠证据，才可能评估新增协议是否划算；本轮不为了并行任务而继续构造实验。

精确源码片段、所有权公式和固定矩形字节量在 `evidence.json`；`extract_evidence.py` 可只读重提取；`artifact_inventory.json` 列文件路径和字节，不做 hash。
