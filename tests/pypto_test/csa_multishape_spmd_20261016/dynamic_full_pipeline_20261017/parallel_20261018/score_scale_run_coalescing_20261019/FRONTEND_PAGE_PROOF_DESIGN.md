# 下一步候选：将页连续性证明移到已有META阶段

当前每叶/每lane证明一遍，B16原输入虽正确但回退0.93%。尚不能仅凭此结果
断言证明开销是根因，但有依据验证把不依赖数学结果的检查移出Score关键路径。

vector_schedule当前在第一名没有HC M8条带的AIV上生成META；T很大时使用
完成自身widen后的47号。该阶段已与HC投影重叠，页表/长度/容量也已可读。
可在同一发布点之前，为每个活动请求证明整个实际压缩历史的正/逆序连续性。
读取页表、Q/L和shape，不读cache payload，不等待Indexer量化数值。

建议复用group_offsets的动态Tensor，在现有scale链表和4*T项描述符之后
追加2*B个INT32（first_page、direction）。旧分区不挪动，不新增Tensor根、
Host回读或task。唯一META写者逐请求覆盖两个字段，inactive或证明失败写0
方向；PHASE_META按既有DCCI/DSB协议发布，Score消费者复用已有获取关系。

整个请求连续才快读，否则保留逐页gather，比逐叶证明更保守。页检查按1024
页分块，保持每块有效列贯穿广播和归约；只能在<2^24域用FP32精确比较。
两条AIV及该请求的全部叶复用结果，不在Score中再做Vector证明。尾页和
逆序TGATHERB读取仍用当前已完整CPU验证的路径。未来实现需更新binding
容量、META调用和Score读取，并重审scratch/布局合同。

这是待实现方案，不是已验证优化。必须核对META没有推迟其它消费者，并用
同轮完整L1及所有冻结门槛判定。当前Score候选不因这个设想提前采用。
