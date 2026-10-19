# Score区间正式采用后的关键路径复核

正式本地提交13259f92，使用完整source_after_score_split；不再以E49作为
当前候选基线。B16/B28都由新policy启用SPLIT_SCORE_ROWS，其它策略不改。

本轮复用已有72角色阶段时戳与Indexer READY/ACK诊断，未修改探针、通知
协议、缓存维护、算子数学或生产代码。现有真实生成C++先作插桩适配审计，
两档各8处原FFTS等待站点全部匹配。阶段时戳与Score背压在同次采集中导出，
每核独占原512B信号槽的第2/3条cache行，不额外增加被测wait周围的barrier。

目的：新Score采用后重新确定前段、Score、Top-K、Attention及输出链的
最长路径，判断READY/ACK主导关系是否改变，再决定一个有>=20微秒潜力的
候选。旧E49全核时间只作历史，不与新带探针时长相减宣称正式收益。

每档只跑一次完整CSA诊断并作输出/Top-K/score/cache/state/保护/代际检查，
不重新全量七档性能/metadata/padding/图回归。插桩结果仅用于归因，最终
候选仍须同轮三方五窗无插桩完整L1门槛。当前无新性能或瓶颈结论。

本地采用提交13259f92已完成，未push。新诊断任务：
- 128K/B16：task_20261020_061948_97386219442，current_score_pressure_h131072_b16_v1。
- 128K/B28：task_20261020_061948_97392211595，current_score_pressure_h131072_b28_v1。

两任务已运行，result中的新SPLIT_SCORE_ROWS均为true；等待终态，不预记通过。

## 诊断终态及逐核结论

两档已completed/exit0，完整单次输出/Top-K/score/cache/state/保护及72角色
信号检查通过。新flag=true且全部原始数据保留，汇总CURRENT_SCORE_PRESSURE_ANALYSIS。
扣除外层GM等待后的Score AIC/AIV中位：B16 192.64/181.67、B28 329.82/324.70μs；
Attention为153.47/158.41、304.19/304.77μs。均含内部等待/流水/探针，不是
引擎独占时间。B16 ACK/READY中位14.65/57.27、B28为31.77/113.09μs。

进一步逐核发现B28最长Score的物理核6在KV后才迟到：KV_PROJECT385.82，
Score前端点加GM累计等待估算起点399.24，而多数空KV核约345～346μs。
该核的两个AIV估算已在382.24/382.60就绪，累计ACK只有13.06μs；因此
把本核Query从Score前移走不能消除它的迟到。这也是不重试旧DQ穿插的
当前依据，而不只是机械沿用历史负例。估算起点不是新增的直接硬件探针。

下一候选只置换24个KV逻辑worker，见PREFIX_KV_BALANCE_DESIGN.md。KV
数学和Score叶/根/所有权不变；保留新Score为参考，不再对E49累计收益。
