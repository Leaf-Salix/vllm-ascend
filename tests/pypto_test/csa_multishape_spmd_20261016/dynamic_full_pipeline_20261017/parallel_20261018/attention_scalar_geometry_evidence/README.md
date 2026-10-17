# Attention 本核四 Scalar 几何惰性缓存

本轮选择且只实现这一候选：同一 AIV 在一次 Attention helper 内，对当前 gather query 的 request/position/active 只计算一次。**源码和纯 Python 同源模型 PASS；候选已冻结，未在本私有目录进行 PyPTO 编译或设备执行，没有性能结论。** 主会话已复制到 `parallel_20261018/attention_scalar_geometry_evidence`，后续唯一完整 CPU 和设备验证由主会话维护，不在此重复。

## 基线与精确改动

`source_reference/` 与 `source_candidate/` 各 33 份源均来自正式 `66882634890af3907d8553d91ed4fab28b4fd631`。仅 [attention_stream_stages.py](source_candidate/attention_stream_stages.py) 改动；不叠加 Cube 16-owner 或 Vector 单-owner 等待候选，不修改共享源码、依赖或安装。[candidate.diff](candidate.diff) 保留完整差异，[prepare_source.py](prepare_source.py) 可复现冻结和修改，已有输出时拒绝覆盖。

当前原函数每个有效 tick 调用 `attention_query_geometry`。该函数默认经 `query_request` 扫描本轮全部请求，并检查唯一 owner、Q 边界、L 及 causal position。五个块属于同一 query，输入在本次 helper 内保持只读，故重复得到同一结果。

候选在 [源码第 267 行](source_candidate/attention_stream_stages.py:267) 增加四个循环携带 Scalar：`geometry_query=-1`、`geometry_owner=-1`、`geometry_position=0`（均 INDEX），`geometry_active=0`（INT32）。它们与现有四个数学 Tile 一起进入 tick 的 init/yield。只有原 `qk_tick<qk_block_count && mask[qk_t,qk_sb]>0` 分支内，且 `qk_t != geometry_query_iter`，才调用**同一原函数**刷新；后续块复用结果。没有追加读取 mask、GM descriptor、Tile、root 参数或 phase。

## 边界与所有权证明

- **首块无效**：不能只在 `sb==0` 刷新。候选以 query tag 判断，首个有效压缩块自然完成初始化。
- **五块全空**：原路径本来不读几何，候选也不读。上一 query tag 暂留无害；下一有效 query 的 tag 不同，必须刷新。
- **active=0**：dummy、陈旧 causal 或多重请求覆盖的返回值也缓存，并更新 tag。后续块仍只写原有零 KV，不因缓存跳过 guard 或把失活变为活动。
- **跨 query / 尾 wave**：`qk_t=core+(tick//5)*24` 随当前 gather 波次变化，实际 T 来自原 device Tensor shape；没有 B/S 白名单或固定 query 数。
- **最后三拍 drain**：原真实 tick 条件已失败，候选保持状态、不读可能越界的 qk_t，也不新增事件。
- **PV 晚三拍**：缓存服务当前 gather，数学 m/l/oi 服务较晚的 PV；旧 query 末块发布时，gather 可能已进入下一 query。因此 [源码末尾](source_candidate/attention_stream_stages.py:535) 只把几何状态原样携带，原 PV reset 不清几何 tag。
- **每次调用 / 图重放**：四条初始化位于 helper 入口，下一调用即使 T 和地址不变而 Q/L/positions 已更新，也重新扫描；没有进程级或 GM 持久缓存。
- **跨核与别名**：每个 AIV 各有自己的四个 Scalar，两 lane 不共享它们，未新增 writer 或共享缓冲。保留入口原 plan/业务获取维护和全部 FFTS。该证明依赖原合法调用合同：Q/L/positions 与页表属于同批数据，helper 内只读、不得与输出非法重叠；若调用过程中外部并发修改 metadata，本来就不满足 plan 与 gather 一致的合同。缓存不依赖 A3 Scalar 跨核 cache coherence。

`attention_query_geometry` 和 `request_geometry.py`、两个 schedule、binding、entry 均逐字保持基线。页表列、负页、物理容量及稀疏 index guard、零 KV、SWA run、压缩逐行搬运、三槽生命周期、READY/ACK、drain、数学舍入和发布均未改变。

## 真正可省工作与成本边界

若某 query 的 k 个块有效（0≤k≤5），每 lane 的 geometry 调用由 k 变为 `1[k>0]`。全五块有效最多减少 4 次扫描。全活动 B16/T96 的具体源码量：每次扫描 16 次 L 和 32 次 Q，再读一次 position 和 owner L，共 50 次逻辑 Scalar 读取；每 AIV 四条 query，20 次 geometry→4 次，约 1000→200 次逻辑读取。全 48 个 AIV 合计少 38400 次**源码读取执行**。这不是 HBM 事务计数，也不能换算成微秒。

代价是每个有效块一个 tag 比较及四个跨 tick Scalar 值；后端可能增加寄存器压力、spill 或改变局部调度。源码没有新 Tile 不等于实际 allocator/机器码一定不变，必须以后续完整 CPU 产物为准。128K/B16 旧插桩中压缩 gather 逐核累计中位 64.76/62.89/63.80 µs、AIC KV wait 114.12/108.97/116.65 µs，只能说明值得审查此推进链，不能当本候选可省时长。该时钟包含 Scalar 发射、几何、地址、清零、搬运及依赖，未单独分解 geometry 成本。

## 历史排重与未选方向

核对 MASTER、动态 ATTEMPTS、`QK_QUERY_METADATA_EXPERIMENT.md`、`qk_query_metadata.py` 和本轮 `attention_aiv_d253_audit/README.md` 后，未发现相同的四 Scalar 惰性缓存已实现/测试：

| 已有方向 | 与本候选的区别 |
| --- | --- |
| 旧每-query QK metadata | 两份 1×512 INT32 地址 Tile，共 4096 B 跨 tick；每 lane 预处理 512 项地址。它不缓存当前动态 Q/L geometry 的四个 Scalar。 |
| 动态 CACHE_QUERY_METADATA | plan 在 GM mask 第5/6列写 owner/position，consumer 每块读 GM 并检查。本候选不扩 plan，metadata 开关仍关闭，原函数只执行一次。 |
| 物理 row plan / UB 地址缓存 / guard 删除 | 针对逐候选页推导和保护，本候选全部保留。 |
| MGATHER / 独立行 Tile / direct-L1 / 跳清零 / PV256 | 改 KV DMA、临时 Tile、事件或数学分块，均不属于本次触点，不重新尝试。 |

“相邻同页的连续行合并搬运”未作为第二候选推进：Top-K 按 score 顺序发布，不能先按索引重排；现有 tick 记录没有候选地址或连续 run 分布。缺少足够长 run 的真实数据时，无法证明合并收益大于 Scalar 探测成本。本次因此选择无需猜测索引连续性的四 Scalar 缓存。

## 已完成审计

[audit_source_and_model.py](audit_source_and_model.py) 只使用 Python 标准库：

1. 校验只改一个文件；去掉确切的四 Scalar 初始化、缓存 if/yield 与循环携带增量后，整个模块 AST（除 docstring）与参考完全一致。
2. 从候选真实 AST 提取缓存控制块/初始化；从原源提取 `request_bounds`、`query_request`、`attention_query_geometry`，移除注解后解释执行，没有手写另一套几何算法。
3. 两条同核 query 的全部 32×32 mask 组合，分别配合 mixed active、dummy 陈旧 position、causal 不一致、请求重叠；另检查 27 个 T 边界及 24 个 core、query 跨波和 drain；同 T 的 A→B→A 更新 Q/L/positions 后重新初始化。

[source_model_audit.json](source_model_audit.json) 为 PASS：7987 次模拟调用、132331 拍、45216 个有效 gather 观察均与原函数一致；候选 geometry 调用 19172 次，每条至少一块有效的 query 恰一次。该混合测试总调用数不是生产工作负载或性能统计。T0 只作无工作/排空模型，Host 仍拒绝非正 T。

可重复只读源码模型（不会编译）：

```bash
python3 /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/attention_scalar_geometry_cache_20261018/audit_source_and_model.py
```

模型没有模拟 DMA/FFTS、设备缓存维护、后端 Scalar 分配或性能；当前审计默认 CACHE_QUERY_METADATA=false。未预记完整后端/设备通过，后续结果另由主会话记录。

## 独立源码审阅

ultra_indexer_policy 已完成只读审阅，未发现反例或阻塞；特别核对 tick7 的 gather 已进入下一 query，而 PV reset 上一 query 时，缓存继续由 geometry_after 单独携带。其 33 源变化范围、Cube/active gather/后续 softmax-PV/IO-FFTS AST 结论与本审计一致。现有 parser/builder 没有 Tile-only 或五项循环状态限制，八项 init/yield 表达有接口依据；这仍不替代 A3 lowering。报告：[README.md](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_attention_scalar_geometry_review_20261018/README.md)。

## 主会话后续：完整CPU通过

主目录唯一cpu_v1已通过；130根、58动态轴、单MIX与sync_start保留。独立生成审阅确认分配与局部事件相同、实际Scalar初始化/携带/guard/drain符合设计。准备两档设备验证，尚无本候选性能结论，详见CPU_REVIEW.json。
